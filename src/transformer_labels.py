ALIASES={"clean":"clean","noise":"noise","mild gibberish":"mild_gibberish","mild_gibberish":"mild_gibberish","word salad":"word_salad","word_salad":"word_salad"}
def normalize_external_label(x):
    x=str(x).lower().strip().replace("-"," ")
    return ALIASES.get(x)
