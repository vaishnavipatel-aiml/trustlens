import numpy as np
import pandas as pd
import uuid
import random
from datetime import datetime, timedelta
from faker import Faker

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
fake = Faker()
Faker.seed(SEED)

N = 20000
FRAUD_RATE = 0.03

DEPARTMENTS = ["Sales", "Engineering", "Marketing", "HR", "Finance", "Ops"]
ROLES = ["IC", "Manager", "Director", "VP"]
CATEGORIES = ["Meals", "Travel", "Software", "Supplies", "Entertainment", "Training"]

CATEGORY_AMOUNT = {
    "Meals": (60, 0.6),
    "Travel": (450, 0.8),
    "Software": (200, 1.0),
    "Supplies": (80, 0.7),
    "Entertainment": (150, 0.9),
    "Training": (600, 0.9),
}

VENDOR_BY_CATEGORY = {
    "Meals": ["Chipotle", "Local Bistro", "Starbucks", "Delta SkyClub"],
    "Travel": ["Delta", "United", "Marriott", "Hertz"],
    "Software": ["AWS", "GitHub", "Slack", "Notion", "Figma"],
    "Supplies": ["Staples", "Amazon Business", "Uline"],
    "Entertainment": ["Topgolf", "Eventbrite", "AMC"],
    "Training": ["Coursera", "O'Reilly", "Udemy Business"],
}

APPROVED_VENDORS = set(v for vs in VENDOR_BY_CATEGORY.values() for v in vs)

MEMO_TEMPLATES = {
    "Meals": ["Team lunch with {n} people", "Client dinner", "Coffee with candidate", "Working lunch"],
    "Travel": ["Flight to {city}", "Hotel for conference", "Rental car for client visit", "Airport parking"],
    "Software": ["Monthly subscription", "License renewal", "Dev tool for team", "Cloud usage"],
    "Supplies": ["Office supplies", "Printer toner", "Team snacks", "Whiteboard markers"],
    "Entertainment": ["Team offsite activity", "Client entertainment", "Team celebration", "Networking event"],
    "Training": ["Online course", "Conference ticket", "Certification exam", "Workshop fee"],
}


def work_hour_datetime():
    base = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 364))
    if base.weekday() >= 5:
        base -= timedelta(days=base.weekday() - 4)
    hour = int(np.clip(np.random.normal(14, 2.5), 8, 20))
    minute = random.randint(0, 59)
    return base.replace(hour=hour, minute=minute)


def base_claim():
    dept = random.choices(DEPARTMENTS, weights=[0.30, 0.25, 0.15, 0.10, 0.10, 0.10])[0]
    role = random.choices(ROLES, weights=[0.55, 0.28, 0.13, 0.04])[0]
    cat = random.choice(CATEGORIES)
    mu, sigma = CATEGORY_AMOUNT[cat]
    role_mult = {"IC": 1.0, "Manager": 1.3, "Director": 1.8, "VP": 2.5}[role]
    amount = round(np.random.lognormal(np.log(mu * role_mult), sigma), 2)
    vendor = random.choice(VENDOR_BY_CATEGORY[cat])
    memo = random.choice(MEMO_TEMPLATES[cat]).format(n=random.randint(2, 8), city=fake.city())
    return {
        "claim_id": str(uuid.uuid4())[:8],
        "employee_id": f"E{random.randint(1000, 1500)}",
        "department": dept,
        "role_level": role,
        "submit_datetime": work_hour_datetime(),
        "expense_date": datetime(2024, 1, 1) + timedelta(days=random.randint(0, 364)),
        "amount": amount,
        "category": cat,
        "vendor": vendor,
        "memo": memo,
        "is_fraud": 0,
        "fraud_type": "none",
    }


def inject_fraud(claim, pattern):
    if pattern == "split_invoice":
        claim["amount"] = round(claim["amount"] / random.uniform(2.0, 3.0), 2)
        claim["memo"] = claim["memo"] + " (partial)"
    elif pattern == "weekend_luxury":
        claim["submit_datetime"] = claim["submit_datetime"].replace(hour=23)
        claim["amount"] *= random.uniform(3.0, 6.0)
        claim["category"] = "Entertainment"
        claim["vendor"] = random.choice(["Michelin Star Bistro", "Private Yacht Charter", "Luxury Spa"])
    elif pattern == "round_number":
        claim["amount"] = float(random.choice([500, 1000, 1500, 2000, 2500]))
    elif pattern == "vendor_mismatch":
        claim["vendor"] = random.choice(["Unknown Vendor LLC", "Cash Advance", "Misc Services"])
        claim["memo"] = "Miscellaneous"
    claim["is_fraud"] = 1
    claim["fraud_type"] = pattern
    return claim


def generate(n=N, fraud_rate=FRAUD_RATE):
    rows = []
    n_fraud = int(n * fraud_rate)
    fraud_indices = set(np.random.choice(n, n_fraud, replace=False))
    patterns = ["split_invoice", "weekend_luxury", "round_number", "vendor_mismatch", "duplicate"]
    for i in range(n):
        c = base_claim()
        if i in fraud_indices:
            c = inject_fraud(c, random.choice(patterns))
        rows.append(c)
    df = pd.DataFrame(rows)
    dups = df[df["fraud_type"] == "duplicate"].copy()
    if len(dups) > 0:
        dups["claim_id"] = [str(uuid.uuid4())[:8] for _ in range(len(dups))]
        df = pd.concat([df, dups], ignore_index=True)
    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv("data/expenses.csv", index=False)
    print(df["fraud_type"].value_counts())
    print(f"Total rows: {len(df)} | Fraud rate: {df['is_fraud'].mean():.3%}")
    print("Saved to data/expenses.csv")
    