use std::collections::HashMap;
use std::fs::File;
use std::io;

use serde::{Deserialize, Serialize};
use serde_json::{from_reader, to_writer, to_writer_pretty};
use unicode_segmentation::UnicodeSegmentation;

const DEFAULT_FILE_PATH: &str = "model.json";
const INITIAL_RATING: f32 = 0.5;
const SPAM_PROB_THRESHOLD: f32 = 0.8;

#[derive(Debug, Default, Serialize, Deserialize)]
struct Counter {
    ham: u32,
    spam: u32,
}

/// A bayesian spam classifier.
#[derive(Default, Debug, Deserialize, Serialize)]
#[serde(from = "ClassifierSerialized")]
pub struct Classifier {
    token_table: HashMap<String, Counter>,
    #[serde(skip)]
    spam_total_count: u32,
    #[serde(skip)]
    ham_total_count: u32,
}

/// The classifier model as it is serialized to disk.
///
/// Does not include the `spam_total_count` and `ham_total_count` fields which
/// can be recomputed from `token_table`.
#[derive(Deserialize, Serialize)]
struct ClassifierSerialized {
    token_table: HashMap<String, Counter>,
}

impl std::convert::From<ClassifierSerialized> for Classifier {
    fn from(c: ClassifierSerialized) -> Self {
        let spam_total_count = c.token_table.values().map(|x| x.spam).sum();
        let ham_total_count = c.token_table.values().map(|x| x.ham).sum();
        Self {
            token_table: c.token_table,
            spam_total_count,
            ham_total_count,
        }
    }
}

impl Classifier {
    /// Build a new classifier with an empty model.
    pub fn new() -> Self {
        Default::default()
    }

    /// Build a new classifier with a pre-trained model loaded from `file`.
    pub fn new_from_pre_trained(file: &mut File) -> Result<Self, io::Error> {
        let pre_trained_model = from_reader(file)?;
        Ok(pre_trained_model)
    }

    /// Save the classifier to `file` as JSON.
    /// The JSON will be pretty printed if `pretty` is `true`.
    pub fn save(&self, file: &mut File, pretty: bool) -> Result<(), io::Error> {
        if pretty {
            to_writer_pretty(file, &self)?;
        } else {
            to_writer(file, &self)?;
        }
        Ok(())
    }

    /// Split `msg` into a list of words.
    fn load_word_list(msg: &str) -> Vec<String> {
        let word_list = msg.unicode_words().collect::<Vec<&str>>();
        word_list.iter().map(|word| word.to_string()).collect()
    }

    /// Train the classifier with a spam `msg`.
    pub fn train_spam(&mut self, msg: &str) {
        for word in Self::load_word_list(msg) {
            let counter = self.token_table.entry(word).or_default();
            counter.spam += 1;
            self.spam_total_count += 1;
        }
    }

    /// Train the classifier with a ham `msg`.
    pub fn train_ham(&mut self, msg: &str) {
        for word in Self::load_word_list(msg) {
            let counter = self.token_table.entry(word).or_default();
            counter.ham += 1;
            self.ham_total_count += 1;
        }
    }

    /// Return the total number of spam in token table.
    fn spam_total_count(&self) -> u32 {
        self.spam_total_count
    }

    /// Return the total number of ham in token table.
    fn ham_total_count(&self) -> u32 {
        self.ham_total_count
    }

    /// Compute the probability of each word of `msg` to be part of a spam.
    fn rate_words(&self, msg: &str) -> Vec<f32> {
        Self::load_word_list(msg)
            .into_iter()
            .map(|word| {
                // If word was previously added in the model
                if let Some(counter) = self.token_table.get(&word) {
                    // If the word has only been part of spam messages,
                    // assign it a probability of 0.99 to be part of a spam
                    if counter.spam > 0 && counter.ham == 0 {
                        return 0.99;
                    // If the word has only been part of ham messages,
                    // assign it a probability of 0.01 to be part of a spam
                    } else if counter.spam == 0 && counter.ham > 0 {
                        return 0.01;
                    // If the word has been part of both spam and ham messages,
                    // calculate the probability to be part of a spam
                    } else if self.spam_total_count() > 0 && self.ham_total_count() > 0 {
                        let ham_prob = (counter.ham as f32) / (self.ham_total_count() as f32);
                        let spam_prob = (counter.spam as f32) / (self.spam_total_count() as f32);
                        return (spam_prob / (ham_prob + spam_prob)).max(0.01);
                    }
                }
                // If word was never added to the model,
                // assign it an initial probability to be part of a spam
                INITIAL_RATING
            })
            .collect()
    }

    /// Compute the spam score of `msg`.
    /// The higher the score, the stronger the liklihood that `msg` is a spam is.
    pub fn score(&self, msg: &str) -> f32 {
        // Compute the probability of each word to be part of a spam
        let ratings = self.rate_words(msg);

        let ratings = match ratings.len() {
            // If there are no ratings, return a score of 0
            0 => return 0.0,
            // If there are more than 20 ratings, keep only the 10 first
            // and 10 last ratings to calculate a score
            x if x > 20 => {
                let length = ratings.len();
                let mut ratings = ratings;
                ratings.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                [&ratings[..10], &ratings[length - 10..]].concat()
            }
            // In all other cases, keep ratings to calculate a score
            _ => ratings,
        };

        // Combine individual ratings
        let product: f32 = ratings.iter().product();
        let alt_product: f32 = ratings.iter().map(|x| 1.0 - x).product();
        product / (product + alt_product)
    }

    /// Identify whether `msg` is a spam or not.
    pub fn identify(&self, msg: &str) -> bool {
        self.score(msg) > SPAM_PROB_THRESHOLD
    }
}

/// Compute the spam score of `msg`, based on a pre-trained model.
/// The higher the score, the stronger the liklihood that `msg` is a spam is.
pub fn score(msg: &str) -> Result<f32, io::Error> {
    let mut file = File::open(DEFAULT_FILE_PATH)?;
    Classifier::new_from_pre_trained(&mut file).map(|classifier| classifier.score(msg))
}

/// Identify whether `msg` is a spam or not, based on a pre-trained model.
pub fn identify(msg: &str) -> Result<bool, io::Error> {
    let score = score(msg)?;
    let is_spam = score > SPAM_PROB_THRESHOLD;
    Ok(is_spam)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        // Create a new classifier with an empty model
        let mut classifier = Classifier::new();

        // Train the classifier with a new spam example
        let spam = "Don't forget our special promotion: -30% on men shoes, only today!";
        classifier.train_spam(spam);

        // Train the classifier with a new ham example
        let ham = "Hi Bob, don't forget our meeting today at 4pm.";
        classifier.train_ham(ham);

        // Identify a typical spam message
        let spam = "Lose up to 19% weight. Special promotion on our new weightloss.";
        let is_spam = classifier.identify(spam);
        assert!(is_spam);

        // Identify a typical ham message
        let ham = "Hi Bob, can you send me your machine learning homework?";
        let is_spam = classifier.identify(ham);
        assert!(!is_spam);
    }

    #[test]
    fn test_new_unicode() {
        // Create a new classifier with an empty model
        let mut classifier = Classifier::new();

        // Train the classifier with a new spam example
        let spam = "Bon plan pour Nöel: profitez de -50% sur le 2ème article.";
        classifier.train_spam(spam);

        // Train the classifier with a new ham example
        let ham = "Vous êtes tous cordialement invités à notre repas de Noël.";
        classifier.train_ham(ham);

        // Identify a typical spam message
        let spam = "Préparez les fêtes de Nöel: 1 article offert!";
        let is_spam = classifier.identify(spam);
        assert!(is_spam);

        // Identify a typical ham message
        let ham = "Pourras-tu être des nôtres pour le repas de Noël?";
        let is_spam = classifier.identify(ham);
        assert!(!is_spam);
    }

    #[test]
    fn test_new_from_pre_trained() -> Result<(), io::Error> {
        // Identify a typical spam message
        let spam = "Lose up to 19% weight. Special promotion on our new weightloss.";
        let is_spam = identify(spam)?;
        assert!(is_spam);

        // Identify a typical ham message
        let ham = "Hi Bob, can you send me your machine learning homework?";
        let is_spam = identify(ham)?;
        assert!(!is_spam);

        Ok(())
    }
}
