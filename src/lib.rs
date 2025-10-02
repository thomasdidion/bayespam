//! # bayespam
//!
//! A simple bayesian spam classifier.
//!
//! ## About
//!
//! Bayespam is inspired by [Naive Bayes classifiers](https://en.wikipedia.org/wiki/Naive_Bayes_spam_filtering), a popular statistical technique of e-mail filtering.
//!
//! Here, the message to be identified is cut into simple words, also called tokens.
//! That are compared to all the corpus of messages (spam or not), to determine the frequency of different tokens in both categories.
//!
//! A probabilistic formula is used to calculate the probability that the message is a spam.
//! When the probability is high enough, the classifier categorizes the message as likely a spam, otherwise as likely a ham.
//! The probability threshold is fixed at 0.8 by default.
//!
//! ## Usage
//!
//! Add to your `Cargo.toml` manifest:
//!
//! ```ini
//! [dependencies]
//! bayespam = "1.1.0"
//! ```
//!
//! ### Use a pre-trained model
//!
//! Add a `model.json` file to your **package root**.
//! Then, you can use it to **score** and **identify** messages:
//!
//! ```
//! use bayespam::classifier;
//!
//! fn main() -> Result<(), std::io::Error> {
//!     // Identify a typical spam message
//!     let spam = "Lose up to 19% weight. Special promotion on our new weightloss.";
//!     let is_spam = classifier::identify(spam)?;
//!     assert!(is_spam);
//!
//!     // Identify a typical ham message
//!     let ham = "Hi Bob, can you send me your machine learning homework?";
//!     let is_spam = classifier::identify(ham)?;
//!     assert!(!is_spam);
//!
//!     Ok(())
//! }
//! ```
//!
//! ### Train your own model
//!
//! You can train a new model from scratch:
//!
//! ```
//! use bayespam::classifier::Classifier;
//!
//! fn main() -> Result<(), std::io::Error> {
//!     // Create a new classifier with an empty model
//!     let mut classifier = Classifier::new();
//!
//!     // Train the classifier with a new spam example
//!     let spam = "Don't forget our special promotion: -30% on men shoes, only today!";
//!     classifier.train_spam(spam);
//!
//!     // Train the classifier with a new ham example
//!     let ham = "Hi Bob, don't forget our meeting today at 4pm.";
//!     classifier.train_ham(ham);
//!
//!     // Identify a typical spam message
//!     let spam = "Lose up to 19% weight. Special promotion on our new weightloss.";
//!     let is_spam = classifier.identify(spam);
//!     assert!(is_spam);
//!
//!     // Identify a typical ham message
//!     let ham = "Hi Bob, can you send me your machine learning homework?";
//!     let is_spam = classifier.identify(ham);
//!     assert!(!is_spam);
//! }
//! ```

pub mod classifier;
