"""
Belief stimuli for epistemic vs. non-epistemic belief differentiation experiments.

Each belief is categorized as:
- epistemic: evidence-based, truth-tracking, factual (can be verified true/false)
- non_epistemic: value-based, desire-based, faith-based, normative (not truth-apt in same way)
"""

EPISTEMIC_BELIEFS = [
    # Science / empirical facts
    "the Earth orbits the Sun",
    "water boils at 100 degrees Celsius at sea level",
    "antibiotics are effective against bacterial infections but not viral infections",
    "the speed of light in a vacuum is approximately 299,792 kilometers per second",
    "DNA carries the genetic instructions for the development of living organisms",
    # History / established facts
    "the Roman Empire fell in 476 AD",
    "the first human Moon landing occurred in 1969",
    "the printing press was invented by Johannes Gutenberg around 1440",
    "World War II ended in 1945",
    "the Great Wall of China was built over many centuries starting from the 7th century BC",
    # Mathematics / logical truths
    "the sum of angles in a triangle equals 180 degrees in Euclidean geometry",
    "prime numbers have exactly two factors: 1 and themselves",
    "the Pythagorean theorem holds for right triangles",
    "there are infinitely many prime numbers",
    "pi is an irrational number",
    # Everyday empirical claims
    "regular exercise is associated with lower risk of heart disease",
    "sleep deprivation impairs cognitive function",
    "smoking increases the risk of lung cancer",
    "the Amazon rainforest produces a significant portion of the world's oxygen",
    "human body temperature averages around 37 degrees Celsius",
]

NON_EPISTEMIC_BELIEFS = [
    # Moral / ethical values
    "honesty is the most important virtue",
    "every person deserves to be treated with dignity and respect",
    "it is wrong to cause unnecessary suffering to animals",
    "freedom of speech should be protected even when the speech is offensive",
    "the death penalty is morally wrong",
    # Personal values / life philosophy
    "hard work is the key to success in life",
    "family is more important than career",
    "money cannot buy happiness",
    "it is better to have loved and lost than never to have loved at all",
    "people should always follow their passion",
    # Aesthetic / preference beliefs
    "classical music is the highest form of artistic expression",
    "nature is more beautiful than any human creation",
    "simplicity is the ultimate sophistication",
    "poetry captures the human experience better than prose",
    "handwritten letters are more meaningful than digital messages",
    # Faith / spiritual / cultural beliefs
    "everything happens for a reason",
    "there is an inherent purpose to human existence",
    "kindness is always rewarded in the end",
    "the universe tends toward justice over time",
    "wisdom is more valuable than knowledge",
]

# 10 statements that can be framed with different verbs (know/believe/value)
VERB_SENSITIVITY_STATEMENTS = [
    ("equality among people", "epistemic_ambiguous"),
    ("the importance of education", "non_epistemic"),
    ("climate change is caused by human activity", "epistemic"),
    ("democracy is the best form of government", "non_epistemic"),
    ("evolution through natural selection", "epistemic"),
    ("the value of forgiveness", "non_epistemic"),
    ("the germ theory of disease", "epistemic"),
    ("justice will prevail", "non_epistemic"),
    ("the effectiveness of vaccination", "epistemic"),
    ("loyalty to one's community", "non_epistemic"),
]

# Counterevidence for belief revision experiment
COUNTEREVIDENCE = {
    # Epistemic beliefs - factual counterevidence
    "the Earth orbits the Sun": "New astronomical observations suggest the Sun actually orbits the Earth in a geocentric model that better explains certain anomalies.",
    "water boils at 100 degrees Celsius at sea level": "Recent precise measurements indicate water actually boils at 99.97 degrees Celsius at standard pressure.",
    "antibiotics are effective against bacterial infections but not viral infections": "A new study claims that certain antibiotics show significant antiviral properties.",
    "the speed of light in a vacuum is approximately 299,792 kilometers per second": "A new experiment reports measuring light traveling at 300,100 km/s in vacuum.",
    "DNA carries the genetic instructions for the development of living organisms": "Researchers propose that RNA, not DNA, is the primary carrier of hereditary information.",
    "the Roman Empire fell in 476 AD": "Historians now argue the Roman Empire persisted in meaningful form until at least 1453.",
    "the first human Moon landing occurred in 1969": "Newly declassified documents suggest the first Moon landing actually happened in 1968 during a secret mission.",
    "the printing press was invented by Johannes Gutenberg around 1440": "Archaeological evidence suggests movable type printing was invented in Korea centuries before Gutenberg.",
    "World War II ended in 1945": "Some historians argue WWII effectively ended in 1944 when the outcome became inevitable.",
    "the Great Wall of China was built over many centuries starting from the 7th century BC": "New archaeological dating suggests the earliest sections date only from the 3rd century BC.",
    "the sum of angles in a triangle equals 180 degrees in Euclidean geometry": "A mathematician claims to have found a counterexample in Euclidean geometry.",
    "prime numbers have exactly two factors: 1 and themselves": "A paper proposes redefining primes to include 1 as a prime number.",
    "the Pythagorean theorem holds for right triangles": "A researcher claims the theorem fails for certain degenerate right triangles.",
    "there are infinitely many prime numbers": "A controversial proof suggests the set of primes is actually finite but extremely large.",
    "pi is an irrational number": "A computation claims to have found a repeating pattern in pi's decimal expansion.",
    "regular exercise is associated with lower risk of heart disease": "A large new meta-analysis finds no significant association between exercise and heart disease risk.",
    "sleep deprivation impairs cognitive function": "A study reports that sleep deprivation actually enhances certain types of creative thinking.",
    "smoking increases the risk of lung cancer": "A study in a specific population found no link between moderate smoking and lung cancer.",
    "the Amazon rainforest produces a significant portion of the world's oxygen": "Scientists now say the Amazon is actually carbon-neutral and produces negligible net oxygen.",
    "human body temperature averages around 37 degrees Celsius": "Recent large-scale studies show human body temperature has declined to an average of 36.6°C.",
    # Non-epistemic beliefs - values-based counterarguments
    "honesty is the most important virtue": "Philosophers argue that compassion, not honesty, should be the primary virtue since honest cruelty causes more harm than compassionate deception.",
    "every person deserves to be treated with dignity and respect": "Some philosophers argue that respect must be earned through one's actions and character, not given automatically.",
    "it is wrong to cause unnecessary suffering to animals": "Utilitarian thinkers argue that some animal suffering is justified when it leads to greater human benefit.",
    "freedom of speech should be protected even when the speech is offensive": "Legal scholars argue unrestricted speech causes measurable harm to vulnerable groups that outweighs its benefits.",
    "the death penalty is morally wrong": "Proponents cite studies suggesting the death penalty deters murder and thus saves innocent lives.",
    "hard work is the key to success in life": "Sociological research shows that social connections and luck are far stronger predictors of success than effort.",
    "family is more important than career": "Studies show people who prioritize career often report higher life satisfaction and self-actualization.",
    "money cannot buy happiness": "Research consistently shows income increases correlate with happiness gains even above $75,000.",
    "it is better to have loved and lost than never to have loved at all": "Psychological research shows the grief from lost love causes lasting trauma that may outweigh the initial joy.",
    "people should always follow their passion": "Career experts argue that skill development and market demand matter more than passion for career satisfaction.",
    "classical music is the highest form of artistic expression": "Ethnomusicologists argue this view is Eurocentric and that many musical traditions are equally complex and expressive.",
    "nature is more beautiful than any human creation": "Art critics point to mathematical perfection in architecture and digital art that surpasses natural forms.",
    "simplicity is the ultimate sophistication": "Engineers and scientists argue that complexity is necessary for solving real-world problems and simplicity is often naive.",
    "poetry captures the human experience better than prose": "Literary scholars argue novels and essays provide deeper, more nuanced exploration of human experience.",
    "handwritten letters are more meaningful than digital messages": "Communication researchers find the medium matters less than the content and emotional intention behind the message.",
    "everything happens for a reason": "Physicists point out that quantum randomness means many events are genuinely purposeless and without cause.",
    "there is an inherent purpose to human existence": "Existentialist philosophers argue existence precedes essence and there is no inherent purpose to life.",
    "kindness is always rewarded in the end": "Empirical studies show kind people are frequently exploited and do not always receive reciprocal benefits.",
    "the universe tends toward justice over time": "Historical evidence shows many grave injustices were never corrected and perpetrators faced no consequences.",
    "wisdom is more valuable than knowledge": "In the modern economy, specialized knowledge consistently leads to better outcomes than general wisdom.",
}
