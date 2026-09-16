import random,re,string,pandas as pd
from sklearn.model_selection import train_test_split
from .common import DATA_DIR,LABELS

SEED=42; N=2500
clean=[
"The model achieved strong performance on the validation dataset.",
"Machine learning systems require reliable data preprocessing.",
"The document contains financial results for the previous fiscal year.",
"The experiment evaluates precision, recall, and inference latency.",
"The patient record was updated after the laboratory results were reviewed.",
"The system uses retrieval augmented generation to answer questions.",
"The convolutional network extracts local patterns from the input signal.",
"The database stores customer transactions and product information.",
"The software pipeline converts documents into searchable representations.",
"The research team compared several models using the same test set.",
"The algorithm was evaluated on unseen samples from a separate cohort.",
"The report describes the architecture, training procedure, and evaluation results.",
"The parser extracts paragraphs, tables, headings, and metadata from documents.",
"The proposed method reduces inference latency while preserving classification quality.",
"The retrieval system returns supporting passages before generating an answer."
]
technical=[
"Conv1d kernel_size=5 hidden_size=128 batch_size=64","HG001 HG002 HG005 nanopore sequencing read classification",
"ROC-AUC 0.9029 MCC 0.5034 F1 0.6915","POST /api/v1/documents status_code=200 latency_ms=17",
"learning_rate=0.002 weight_decay=0.0003 warmup_steps=1200","SELECT customer_id, revenue FROM transactions WHERE year=2025",
"https://example.com/api/v1/search?q=transformer","CUDA 12.4 PyTorch 2.5 NVIDIA A100 80GB"
]
pool=["banana","quantum","purple","running","database","window","mathematics","cloud","river","algorithm","chair","oxygen","network","garden","tensor","planet","coffee","gradient","memory","camera","signal","language","forest","protein","market","kernel","document","engine","blue","sequence","processor","mountain","battery","invoice","galaxy","library","temperature","triangle","airport","software","elephant"]
keys=["asdfghjkl","qwertyuiop","zxcvbnm","qazwsxedc","plmoknijb"]

def clean_text(r,i):
    x=r.choice(clean+technical)
    x=r.choice(["","In this experiment, ","According to the report, ","For the benchmark, "])+x
    x+=r.choice([""," The results were reproducible."," The experiment used a fixed random seed."," The result was recorded for later analysis."])
    return x+f" Sample {i}."

def noise(r):
    a=string.ascii_lowercase+string.digits+"!@#$%^&*"
    k=r.randrange(4)
    if k==0:return "".join(r.choice(a) for _ in range(r.randint(10,60)))
    if k==1:return r.choice(keys)+" "+r.choice(keys)
    if k==2:return "".join(r.choice("@#$%&*!?") for _ in range(r.randint(8,30)))
    return " ".join("".join(r.choice(string.ascii_letters) for _ in range(r.randint(2,8))) for _ in range(r.randint(3,10)))

def salad(r):
    return " ".join(r.sample(pool,r.randint(6,15)))

def mild(r,i):
    x=clean_text(r,i)
    reps={"model":"rnodel","system":"systme","data":"dtaa","document":"documnt","performance":"performnce","algorithm":"algortihm","retrieval":"retrival"}
    words=x.split()
    for j,w in enumerate(words):
        key=re.sub("[^a-z]","",w.lower())
        if key in reps and r.random()<.8: words[j]=reps[key]
    if r.random()<.5:
        j=r.randrange(len(words))
        if len(words[j])>5: words[j]=words[j][:2]+words[j][3:]
    if r.random()<.35: words.insert(r.randrange(len(words)),r.choice(keys))
    return " ".join(words)

def unique(label, n, offset, global_seen):
    r = random.Random(SEED + offset)
    s = set()
    i = 0

    print(f"[DATA] Generating {n} unique {label} samples...")

    while len(s) < n:
        if label == "clean":
            x = clean_text(r, i)

        elif label == "noise":
            x = noise(r)

        elif label == "mild_gibberish":
            x = mild(r, i)

        elif label == "word_salad":
            x = salad(r) + f" {i}" if r.random() < 0.3 else salad(r)

        else:
            raise ValueError(label)

        x = re.sub(r"\s+", " ", x).strip()

        # Guarantee uniqueness across the ENTIRE dataset.
        if x and x not in global_seen:
            s.add(x)
            global_seen.add(x)

        i += 1

        if i > n * 500:
            raise RuntimeError(
                f"Could not generate {n} globally unique samples "
                f"for class '{label}'."
            )

    print(
        f"[DATA] {label}: {len(s)} unique samples generated "
        f"after {i} attempts."
    )

    return list(s)


def main():
    print("\n==============================================")
    print("          DATASET GENERATION V3")
    print("==============================================")

    rows = []

    # This set tracks every text generated across ALL classes.
    global_seen = set()

    for j, label in enumerate(LABELS):

        texts = unique(
            label=label,
            n=N,
            offset=j * 100,
            global_seen=global_seen,
        )

        rows += [
            {
                "text": x,
                "label": label,
            }
            for x in texts
        ]

    df = pd.DataFrame(rows)

    print("\n==============================================")
    print("             FINAL DATASET CHECK")
    print("==============================================")

    print("\n[DATA] Class distribution:")
    print(
        df["label"]
        .value_counts()
        .sort_index()
    )

    print("\n[DATA] Total samples:")
    print(len(df))

    duplicate_count = df["text"].duplicated().sum()

    print("\n[DATA] Duplicate texts:")
    print(duplicate_count)

    print("\n[DATA] Unique texts:")
    print(df["text"].nunique())

    # Strong benchmark validation.
    assert len(df) == 10000, \
        f"Expected 10000 rows, got {len(df)}"

    assert duplicate_count == 0, \
        f"Found {duplicate_count} duplicate texts"

    counts = df["label"].value_counts()

    for label in LABELS:
        assert counts[label] == 2500, \
            f"{label} has {counts[label]} samples instead of 2500"

    print("\n[DATA] All dataset checks PASSED.")
    print("[DATA] 10,000 samples")
    print("[DATA] 2,500 samples per class")
    print("[DATA] 0 duplicate texts")

    # Stratified 70/15/15 split.
    train, temp = train_test_split(
        df,
        test_size=0.30,
        stratify=df["label"],
        random_state=SEED,
    )

    val, test = train_test_split(
        temp,
        test_size=0.50,
        stratify=temp["label"],
        random_state=SEED,
    )

    out = DATA_DIR / "benchmark"
    out.mkdir(exist_ok=True)

    train.to_csv(
        out / "train.csv",
        index=False,
    )

    val.to_csv(
        out / "val.csv",
        index=False,
    )

    test.to_csv(
        out / "test.csv",
        index=False,
    )

    print("\n==============================================")
    print("              SPLIT SUMMARY")
    print("==============================================")

    print(f"[DATA] Train      : {len(train)}")
    print(f"[DATA] Validation : {len(val)}")
    print(f"[DATA] Test       : {len(test)}")

    for name, data in [
        ("TRAIN", train),
        ("VALIDATION", val),
        ("TEST", test),
    ]:

        print(f"\n[DATA] {name} distribution:")
        print(
            data["label"]
            .value_counts()
            .sort_index()
        )

    print(
        f"\n[DATA] Benchmark saved to:\n{out}"
    )


if __name__ == "__main__":
    main()
