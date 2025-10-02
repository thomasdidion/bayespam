use bayespam::classifier;

fn main() -> Result<(), std::io::Error> {
    // Identify a typical spam message
    let spam = "Lose up to 19% weight. Special promotion on our new weightloss.";
    let score = classifier::score(spam)?;
    let is_spam = classifier::identify(spam)?;
    println!("{:.4?}", score);
    println!("{:?}", is_spam);

    // Identify a typical ham message
    let ham = "Hi Bob, can you send me your machine learning homework?";
    let score = classifier::score(ham)?;
    let is_spam = classifier::identify(ham)?;
    println!("{:.4?}", score);
    println!("{:?}", is_spam);

    Ok(())
}
