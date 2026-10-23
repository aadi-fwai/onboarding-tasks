#!/usr/bin/env python3
"""Generate the SFT/DPO datasets for the query-domain classifier task.

Classes: math, code, history, finance, other
Output files (this directory):
  train.jsonl     - 200 rows (40/class), chat-format SFT examples
  eval.jsonl      - 50 rows (10/class), held-out for evaluation
  dpo_train.jsonl - 40 chosen/rejected preference pairs for the DPO stage

SFT row format (Fireworks managed SFT):
  {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}
The trainer applies the model's chat template and computes loss on the
assistant turn only (system/user turns are masked).

DPO row format (Fireworks managed DPO):
  {"messages": [{"role": "system", ...}, {"role": "user", ...}],
   "chosen": {"role": "assistant", "content": ...},
   "rejected": {"role": "assistant", "content": ...}}
"""

import json
import os
import random

random.seed(42)

SYSTEM_PROMPT = (
    "You are a query domain classifier. Classify the user's query into exactly one "
    "domain: math, code, history, finance, other. Respond with only a JSON object "
    'of the form {"domain": "<label>"} and nothing else.'
)

DOMAINS = ["math", "code", "history", "finance", "other"]

# Hand-written queries per class, varied in phrasing/length so the model
# learns the task rather than memorizing strings.
QUERIES = {
    "math": [
        "What is the derivative of x^3 + 2x?",
        "Solve for x: 3x + 7 = 22",
        "What is the integral of sin(x) dx?",
        "Compute the probability of rolling two sixes with fair dice.",
        "Find the eigenvalues of the matrix [[2, 1], [1, 2]].",
        "What is 15% of 240?",
        "Prove that the square root of 2 is irrational.",
        "What is the sum of the first 100 positive integers?",
        "Simplify the expression (x^2 - 9) / (x - 3).",
        "How many ways can you choose 3 books from a shelf of 10?",
        "What is the limit of (1 + 1/n)^n as n approaches infinity?",
        "Convert 45 degrees to radians.",
        "What is the greatest common divisor of 48 and 180?",
        "If a triangle has sides 3, 4, and 5, is it a right triangle?",
        "What is the standard deviation of the set {2, 4, 4, 4, 5, 5, 7, 9}?",
        "Differentiate f(x) = e^(2x) * cos(x).",
        "How many prime numbers are there between 1 and 50?",
        "What is the area of a circle with radius 7?",
        "Solve the quadratic equation x^2 - 5x + 6 = 0.",
        "What is the binomial expansion of (a + b)^4?",
        "Calculate the compound interest on $1000 at 5% annually for 3 years.",
        "What is log base 2 of 64?",
        "Find the median of the numbers 12, 7, 3, 9, 15, 1.",
        "What is the dot product of vectors (1, 2, 3) and (4, -5, 6)?",
        "How do you compute the slope of a line given two points?",
        "What is the factorial of 10?",
        "Is 91 a prime number? Explain your reasoning.",
        "What is the volume of a cone with radius 3 and height 8?",
        "Solve the system: 2x + y = 7, x - y = 2.",
        "What is the 10th Fibonacci number?",
        "Evaluate the determinant of a 2x2 matrix [[4, 3], [6, 3]].",
        "What is the expected value of a fair six-sided die roll?",
        "How many edges does a cube have?",
        "What is the cosine of 60 degrees?",
        "Express 0.75 as a fraction in lowest terms.",
        "What is the arithmetic mean of 10, 20, 30, 40?",
        "If y = 4x - 3, what is y when x = 5?",
        "What is the perimeter of a rectangle 8 by 5?",
        "Explain the Pythagorean theorem with an example.",
        "What is the least common multiple of 6, 8, and 12?",
        "How do you convert a decimal to a percentage?",
        "What is the variance of the set {1, 2, 3, 4, 5}?",
        "Simplify sqrt(72).",
        "What is the probability of drawing an ace from a standard deck?",
        "Find the inverse of the function f(x) = 3x + 2.",
        "What is the remainder when 1234 is divided by 7?",
        "How many diagonals does a hexagon have?",
        "What is the radius of a circle with circumference 31.4?",
        "Compute 17 choose 2.",
        "What is the square root of 169?",
    ],
    "code": [
        "Write a Python function to reverse a string.",
        "How do I fix a NullPointerException in Java?",
        "What's the difference between git merge and git rebase?",
        "Explain what a hash map is and its time complexity.",
        "Write a SQL query to find duplicate emails in a users table.",
        "How do you center a div in CSS?",
        "What does the 'this' keyword mean in JavaScript?",
        "Write a regex to validate an email address.",
        "How do I read a file line by line in Python?",
        "What is the difference between a stack and a queue?",
        "Implement binary search in C++.",
        "How do promises work in JavaScript?",
        "What causes a segmentation fault in C?",
        "How do I make an HTTP GET request in Go?",
        "Explain the difference between TCP and UDP.",
        "Write a function to check if a string is a palindrome in JavaScript.",
        "What is dependency injection and why is it useful?",
        "How do I sort a list of dictionaries by a key in Python?",
        "What is the time complexity of quicksort in the worst case?",
        "How do you handle exceptions in Rust?",
        "Write a bash script to count the number of files in a directory.",
        "What is the difference between == and === in JavaScript?",
        "How do I set up a virtual environment in Python?",
        "Explain what recursion is with an example in Python.",
        "How do I connect to a PostgreSQL database using Node.js?",
        "What is a REST API and how does it differ from GraphQL?",
        "Write a JavaScript function to debounce another function.",
        "How do you reverse a linked list in place?",
        "What are Python decorators and how do you write one?",
        "How do I parse JSON in Java?",
        "What is the difference between an abstract class and an interface?",
        "Write a Dockerfile for a simple Flask app.",
        "How do you find the longest common subsequence of two strings?",
        "What does the 'static' keyword mean in Java?",
        "How do I remove duplicates from a list in Python?",
        "Explain how garbage collection works in Go.",
        "What is the difference between let, const, and var?",
        "How do you implement a breadth-first search?",
        "Write a unit test for a function that adds two numbers.",
        "What is the purpose of the 'yield' keyword in Python?",
        "How do I handle CORS errors in an Express server?",
        "What is the difference between HTTP and HTTPS?",
        "Write a function to compute Fibonacci numbers with memoization.",
        "How do I list all running docker containers?",
        "What is tail-call optimization?",
        "How do you check if a binary tree is balanced?",
        "What is the difference between process.env and dotenv in Node?",
        "How do I convert a string to an integer in Rust?",
        "Explain the event loop in JavaScript.",
        "What is the difference between an INNER JOIN and a LEFT JOIN?",
    ],
    "history": [
        "What caused the fall of the Roman Empire?",
        "Who was the first president of the United States?",
        "What year did World War II end?",
        "Explain the significance of the Magna Carta.",
        "What was the Industrial Revolution and when did it begin?",
        "Who led the civil rights movement in the 1960s?",
        "What was the Cold War and which countries were involved?",
        "When did humans first land on the Moon?",
        "What was the French Revolution and what triggered it?",
        "Who wrote the Declaration of Independence?",
        "What was the Silk Road and why was it important?",
        "Explain the causes of World War I.",
        "What was the Renaissance and where did it start?",
        "Who was Genghis Khan and what did he conquer?",
        "What happened during the Boston Tea Party?",
        "When was the printing press invented and by whom?",
        "What was the Great Depression and how did it end?",
        "Who were the main Axis powers in World War II?",
        "What was the Berlin Wall and when did it fall?",
        "Explain the significance of the Battle of Hastings.",
        "Who was Cleopatra and which empire did she rule?",
        "What was the Reformation and who started it?",
        "When did the American Civil War take place?",
        "What was the Ottoman Empire and how long did it last?",
        "Who discovered America in 1492?",
        "What was the Louisiana Purchase?",
        "Explain the role of women in World War II.",
        "What was the Black Death and when did it occur?",
        "Who was Alexander the Great and what did he achieve?",
        "What was the Treaty of Versailles?",
        "When did India gain independence from Britain?",
        "What was the space race between the US and USSR?",
        "Who was Julius Caesar and how did he die?",
        "What was the Enlightenment and who were its key thinkers?",
        "Explain the causes of the Russian Revolution of 1917.",
        "What was the role of the Roman Senate?",
        "When was the United Nations founded?",
        "What was the Meiji Restoration in Japan?",
        "Who was Napoleon Bonaparte and where was he exiled?",
        "What was the Underground Railroad?",
        "Explain the significance of the Battle of Gettysburg.",
        "What was the Byzantine Empire?",
        "When did the Titanic sink and how many people died?",
        "What was the Great Wall of China built to defend against?",
        "Who was Charlemagne and why is he important?",
        "What was the War of 1812 fought over?",
        "Explain the history of the Suez Canal.",
        "What was the Columbian Exchange?",
        "Who was Winston Churchill and what was his role in WWII?",
        "What was the Boston Massacre and why did it matter?",
    ],
    "finance": [
        "What is the difference between a stock and a bond?",
        "How does compound interest work in a savings account?",
        "What is a 401(k) and how does it differ from an IRA?",
        "Explain what a mutual fund is.",
        "What causes inflation in an economy?",
        "How do I create a monthly budget?",
        "What is the difference between a bull market and a bear market?",
        "What is an ETF and how does it differ from a mutual fund?",
        "How does a mortgage amortization schedule work?",
        "What is the Federal Reserve and what does it do?",
        "Explain what diversification means in investing.",
        "What is a credit score and how is it calculated?",
        "What is the difference between gross income and net income?",
        "How do stock splits affect share price?",
        "What is a Roth IRA and who qualifies?",
        "Explain what a hedge fund does.",
        "What is the price-to-earnings ratio and why does it matter?",
        "How does inflation affect the purchasing power of money?",
        "What is an emergency fund and how much should it cover?",
        "What is the difference between a debit card and a credit card?",
        "Explain the concept of dollar-cost averaging.",
        "What is a balance sheet and what does it show?",
        "How do dividends work?",
        "What is the time value of money?",
        "What is capital gains tax and when do you pay it?",
        "Explain what an IPO is.",
        "What is the difference between fixed and variable interest rates?",
        "How does a health savings account (HSA) work?",
        "What is a recession and how is it defined?",
        "What is the difference between revenue and profit?",
        "How do I calculate my net worth?",
        "What is a certificate of deposit (CD)?",
        "Explain what short selling means.",
        "What is the S&P 500 index?",
        "How do exchange-traded treasury bonds work?",
        "What is an annuity and who should consider one?",
        "What is the difference between active and passive investing?",
        "How does a credit card's APR work?",
        "What is a fiduciary financial advisor?",
        "Explain the concept of asset allocation.",
        "What is a money market account?",
        "How do I read a company's income statement?",
        "What is the debt-to-income ratio and why do lenders use it?",
        "What are municipal bonds and what tax advantages do they offer?",
        "How does refinancing a mortgage work?",
        "What is a limit order versus a market order?",
        "Explain what a sovereign wealth fund is.",
        "What is the difference between term and whole life insurance?",
        "How does currency exchange work when traveling abroad?",
        "What is a stock option and how do employees use them?",
    ],
    "other": [
        "What's a good recipe for chocolate chip cookies?",
        "How do I train for a marathon as a beginner?",
        "What are some tips for improving my sleep quality?",
        "Recommend a good book for learning about psychology.",
        "How do I change a flat tire on a car?",
        "What are the best exercises for lower back pain?",
        "How do I start a vegetable garden in my backyard?",
        "What's the best way to learn a new language as an adult?",
        "How do I remove a wine stain from a white shirt?",
        "What should I pack for a two-week trip to Japan?",
        "How do I improve my public speaking skills?",
        "What's a good workout routine for someone who hates the gym?",
        "How do I house-train a new puppy?",
        "What are some easy weeknight dinner ideas?",
        "How do I take care of indoor succulents?",
        "What's the difference between espresso and drip coffee?",
        "How do I negotiate a higher salary at my job?",
        "What are the symptoms of seasonal allergies?",
        "How do I plan a budget-friendly wedding?",
        "What's the best way to learn photography?",
        "How do I unclog a kitchen sink?",
        "What should I look for when buying a used car?",
        "How do I meditate for stress relief?",
        "What are some fun activities for a rainy day with kids?",
        "How do I make homemade bread from scratch?",
        "What's the best way to organize a small closet?",
        "How do I write a good cover letter?",
        "What are the health benefits of drinking green tea?",
        "How do I start running if I've never exercised before?",
        "What's the best time of year to visit Italy?",
        "How do I teach my child to ride a bike?",
        "What are some good podcasts about true crime?",
        "How do I get better at cooking without following recipes?",
        "What's the proper way to set a formal dinner table?",
        "How do I declutter my home in a weekend?",
        "What should I consider before adopting a cat?",
        "How do I stay motivated while working from home?",
        "What's the difference between a latte and a cappuccino?",
        "How do I plan a road trip across the country?",
        "What are some tips for better time management?",
        "How do I care for a bonsai tree?",
        "What's a good way to meet new people in a new city?",
        "How do I improve my credit card rewards strategy?",
        "What are the best stretches to do before a run?",
        "How do I make cold brew coffee at home?",
        "What should I bring to a potluck dinner?",
        "How do I start journaling regularly?",
        "What are some signs of dehydration to watch for?",
        "How do I choose the right running shoes?",
        "What's the best way to store fresh herbs?",
    ],
}


def sft_row(query: str, label: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
            {"role": "assistant", "content": json.dumps({"domain": label})},
        ]
    }


# DPO rejected responses: plausible base-model failure modes.
# 1) wrong label, 2) verbose prose instead of JSON, 3) invalid JSON.
def rejected_for(query: str, label: str, rng: random.Random) -> str:
    mode = rng.choice(["wrong_label", "verbose", "invalid_json"])
    if mode == "wrong_label":
        wrong = rng.choice([d for d in DOMAINS if d != label])
        return json.dumps({"domain": wrong})
    if mode == "verbose":
        return (
            f"This query is best classified as {label}, since it deals with "
            f"questions related to the {label} domain."
        )
    return "{domain: " + label + "}"  # not valid JSON (unquoted key)


def main():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    rng = random.Random(42)

    # Split: first 40 per class -> train, last 10 per class -> eval
    train_rows, eval_rows = [], []
    for label in DOMAINS:
        qs = QUERIES[label]
        assert len(qs) >= 50, f"need >=50 queries for {label}, have {len(qs)}"
        train_rows += [sft_row(q, label) for q in qs[:40]]
        eval_rows += [sft_row(q, label) for q in qs[40:50]]

    rng.shuffle(train_rows)
    rng.shuffle(eval_rows)

    # DPO pairs from a sample of the train split
    dpo_rows = []
    for row in rng.sample(train_rows, 40):
        user = row["messages"][1]["content"]
        chosen = row["messages"][2]["content"]
        label = json.loads(chosen)["domain"]
        dpo_rows.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                "chosen": {"role": "assistant", "content": chosen},
                "rejected": {"role": "assistant", "content": rejected_for(user, label, rng)},
            }
        )

    for name, rows in [("train.jsonl", train_rows), ("eval.jsonl", eval_rows), ("dpo_train.jsonl", dpo_rows)]:
        path = os.path.join(out_dir, name)
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {path}: {len(rows)} rows")


if __name__ == "__main__":
    main()
