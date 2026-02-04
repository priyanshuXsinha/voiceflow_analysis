import pandas as pd
import random
from datetime import datetime, timedelta

# --- CONFIGURATION ---
TOTAL_REVIEWS = 600  # Total number of reviews to generate

# --- BUILDING BLOCKS ---

# 1. Categories
categories = ["Payment & App", "Delivery Service", "Food Quality", "Customer Support"]

# 2. Negative Templates
neg_intros = [
    "I am extremely disappointed.", "Worst service ever.", "Total waste of money.",
    "Unacceptable experience.", "Pathetic app.", "I regret ordering from here.",
    "Do not download this app.", "Frustrated with the service."
]

neg_issues = {
    "Payment & App": [
        "money deducted but order failed",
        "app crashes on checkout",
        "login OTP never received",
        "hidden charges added"
    ],
    "Delivery Service": [
        "delivery was 2 hours late",
        "rider was rude and yelled",
        "order spilled in the bag",
        "tracker showed wrong location"
    ],
    "Food Quality": [
        "food was cold and stale",
        "found a hair in the food",
        "portion size is too small",
        "tasted weird and chemical-like"
    ],
    "Customer Support": [
        "chat support is a bot loop",
        "no one picks up the call",
        "refused refund for missing item",
        "agent was unhelpful"
    ]
}

neg_outros = [
    "Deleting the app now.", "I want a full refund.", "Never ordering again.",
    "Giving 0 stars.", "Beware guys.", "Totally unprofessional."
]

# 3. Positive Templates
pos_intros = [
    "Absolutely loved it!", "Great experience.", "Super fast service.",
    "My go-to app.", "Five stars!", "Highly recommended.", "Smooth interface."
]

pos_features = {
    "Payment & App": [
        "UI is buttery smooth",
        "dark mode looks great",
        "payment was instant",
        "got a great discount coupon"
    ],
    "Delivery Service": [
        "delivery within 20 mins",
        "rider was polite",
        "packaging was secure",
        "contactless delivery followed"
    ],
    "Food Quality": [
        "food was piping hot",
        "taste was authentic",
        "fresh ingredients used",
        "best biryani ever"
    ],
    "Customer Support": [
        "refund processed instantly",
        "support team is very polite",
        "resolved my issue in seconds",
        "very helpful agent"
    ]
}

pos_outros = [
    "Will order again.", "Keep it up!", "Good job team.",
    "Made my dinner special.", "Loyal customer here."
]

# --- GENERATOR LOGIC ---
data = []

start_date = datetime(2025, 1, 1)

for i in range(TOTAL_REVIEWS):
    # 60% Negative, 40% Positive distribution
    sentiment = "Negative" if random.random() < 0.6 else "Positive"

    category = random.choice(categories)
    date_posted = start_date + timedelta(days=random.randint(0, 365))

    if sentiment == "Negative":
        text = (
            f"{random.choice(neg_intros)} "
            f"{random.choice(neg_issues[category])}. "
            f"{random.choice(neg_outros)}"
        )
        rating = random.choice([1, 2])
    else:
        text = (
            f"{random.choice(pos_intros)} "
            f"{random.choice(pos_features[category])}. "
            f"{random.choice(pos_outros)}"
        )
        rating = random.choice([4, 5])

    data.append({
        "review_id": i + 1001,
        "date": date_posted.strftime("%Y-%m-%d"),
        "category": category,
        "rating": rating,
        "sentiment": sentiment,
        "review_text": text
    })

# --- SHUFFLE & SAVE ---
df = pd.DataFrame(data)
df = df.sample(frac=1).reset_index(drop=True)  # Shuffle rows

df.to_csv("reviews.csv", index=False)
print(f"Generated {TOTAL_REVIEWS} realistic reviews in 'reviews.csv'")
print(df.head())
