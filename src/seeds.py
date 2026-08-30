"""Seed loaders for the 4 paper subjects.

Each loader returns a list[(problem, ground_truth)].

For reproducibility:
- All loaders accept `n_problems` and `seed`
- Datasets fetched from HuggingFace are cached locally
- We provide built-in fallback seeds for offline/MVP use

Subjects:
  math      : MATH-500 (or built-in 50 fallback)
  code      : MBPP     (or built-in 30 fallback)
  science   : ScienceQA (or built-in 30 fallback)
  chinese   : CMMLU    (or built-in 20 fallback, in Chinese)
"""
from __future__ import annotations

import logging
import random
from typing import Iterable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Built-in fallback seeds (no internet required)
# ---------------------------------------------------------------------------

MATH_FALLBACK = [
    ("If 3x + 7 = 22, what is x?", "5"),
    ("What is the area of a triangle with base 6 and height 8?", "24"),
    ("Solve for y: 2y - 4 = 10.", "7"),
    ("What is 15% of 80?", "12"),
    ("If a circle has radius 5, what is its circumference? (use π=3.14)", "31.4"),
    ("What is the next term: 2, 4, 8, 16, ?", "32"),
    ("Factor: x^2 - 9", "(x-3)(x+3)"),
    ("If f(x) = 2x + 3, what is f(4)?", "11"),
    ("What is the slope of the line through (1,2) and (3,8)?", "3"),
    ("Solve: log_2(8) = ?", "3"),
    ("What is the derivative of x^3?", "3x^2"),
    ("Evaluate: 2^5", "32"),
    ("Mean of [4, 8, 6, 10, 2]?", "6"),
    ("Median of [3, 1, 7, 5, 9]?", "5"),
    ("What is sin(30°)?", "0.5"),
    ("How many sides does a hexagon have?", "6"),
    ("Solve: |x - 3| = 7. What are the solutions?", "x=10 or x=-4"),
    ("What is the sum of angles in a triangle?", "180"),
    ("Probability of rolling a 6 on a fair die?", "1/6"),
    ("If 5 books cost $40, how much do 8 books cost?", "$64"),
    ("Convert 0.75 to a fraction.", "3/4"),
    ("Simplify: (x^2)(x^3)", "x^5"),
    ("What is the GCD of 12 and 18?", "6"),
    ("LCM of 4 and 6?", "12"),
    ("If a square has perimeter 20, what is its area?", "25"),
    ("What is 7! (7 factorial)?", "5040"),
    ("Volume of a cube with side 3?", "27"),
    ("How many degrees in a right angle?", "90"),
    ("What is √144?", "12"),
    ("Solve: x/4 = 9.", "36"),
    ("Number of diagonals in a pentagon?", "5"),
    ("Sum of first 10 positive integers?", "55"),
    ("What is e^0?", "1"),
    ("Convert 2π radians to degrees.", "360"),
    ("What is 3/4 + 1/8 ?", "7/8"),
    ("Solve: 2(x+3) = 14.", "4"),
    ("Number of permutations of 4 distinct items?", "24"),
    ("If a=2, b=3, what is a^b + b^a?", "17"),
    ("What is the discriminant of x^2 - 5x + 6?", "1"),
    ("Roots of x^2 - 4 = 0?", "x=2 or x=-2"),
    ("If P(A)=0.3, P(B)=0.4, P(A and B)=0.12, are A and B independent?", "Yes"),
    ("Sum of geometric series 1+1/2+1/4+1/8+...?", "2"),
    ("How many primes between 1 and 20?", "8"),
    ("Logarithm: log_10(1000)?", "3"),
    ("Distance between (0,0) and (3,4)?", "5"),
    ("If sin(θ)=3/5, what is cos(θ) (acute)?", "4/5"),
    ("Solve: e^x = 1.", "0"),
    ("Vector dot product (1,2)·(3,4)?", "11"),
    ("Limit of (1 + 1/n)^n as n→∞?", "e"),
    ("Integral of 2x dx?", "x^2 + C"),
]

CODE_FALLBACK = [
    ("Write a Python function `is_palindrome(s)` that returns True if `s` reads the same forward and backward.",
     "def is_palindrome(s): return s == s[::-1]"),
    ("Write a function `sum_list(nums)` that returns the sum of a list of integers.",
     "def sum_list(nums): return sum(nums)"),
    ("Write `fibonacci(n)` returning the n-th Fibonacci number (n>=0).",
     "def fibonacci(n):\n    a,b=0,1\n    for _ in range(n): a,b=b,a+b\n    return a"),
    ("Write `factorial(n)` for non-negative integer n.",
     "def factorial(n):\n    if n<2: return 1\n    return n*factorial(n-1)"),
    ("Write `is_prime(n)` returning True if n is prime.",
     "def is_prime(n):\n    if n<2: return False\n    for i in range(2, int(n**0.5)+1):\n        if n%i==0: return False\n    return True"),
    ("Write `reverse_words(s)` that reverses the order of words in a string.",
     "def reverse_words(s): return ' '.join(s.split()[::-1])"),
    ("Write `count_vowels(s)` returning the number of vowels in s (a,e,i,o,u, case-insensitive).",
     "def count_vowels(s): return sum(c.lower() in 'aeiou' for c in s)"),
    ("Write `merge_sorted(a, b)` to merge two sorted lists into one sorted list.",
     "def merge_sorted(a,b):\n    r=[]; i=j=0\n    while i<len(a) and j<len(b):\n        if a[i]<=b[j]: r.append(a[i]); i+=1\n        else: r.append(b[j]); j+=1\n    return r+a[i:]+b[j:]"),
    ("Write `flatten(nested)` that flattens a list of lists into a single list.",
     "def flatten(nested): return [x for sub in nested for x in sub]"),
    ("Write `gcd(a,b)` using Euclid's algorithm.",
     "def gcd(a,b):\n    while b: a,b=b,a%b\n    return a"),
    ("Write `to_binary(n)` returning binary representation of non-negative integer as string.",
     "def to_binary(n): return bin(n)[2:] if n else '0'"),
    ("Write `most_frequent(lst)` returning the most-frequent element of a list.",
     "def most_frequent(lst):\n    from collections import Counter\n    return Counter(lst).most_common(1)[0][0]"),
    ("Write `is_anagram(s1,s2)` returning True iff s1 and s2 are anagrams.",
     "def is_anagram(s1,s2): return sorted(s1)==sorted(s2)"),
    ("Write `nth_prime(n)` returning the n-th prime (1-indexed).",
     "def nth_prime(n):\n    primes=[]; k=2\n    while len(primes)<n:\n        if all(k%p for p in primes): primes.append(k)\n        k+=1\n    return primes[-1]"),
    ("Write `caesar(s, k)` shifting each letter by k positions (preserve case, leave non-letters).",
     "def caesar(s,k):\n    out=[]\n    for c in s:\n        if c.isupper(): out.append(chr((ord(c)-65+k)%26+65))\n        elif c.islower(): out.append(chr((ord(c)-97+k)%26+97))\n        else: out.append(c)\n    return ''.join(out)"),
    ("Write `chunk(lst, n)` splitting a list into chunks of size n.",
     "def chunk(lst,n): return [lst[i:i+n] for i in range(0,len(lst),n)]"),
    ("Write `unique_preserve_order(lst)` removing duplicates while preserving first-seen order.",
     "def unique_preserve_order(lst):\n    seen=set(); out=[]\n    for x in lst:\n        if x not in seen: seen.add(x); out.append(x)\n    return out"),
    ("Write `power(base, exp)` without using `**`.",
     "def power(b,e):\n    r=1\n    for _ in range(e): r*=b\n    return r"),
    ("Write `count_occurrences(s, sub)` returning how many times sub occurs in s (overlapping not counted).",
     "def count_occurrences(s,sub): return s.count(sub)"),
    ("Write `mean(nums)` returning the arithmetic mean of a non-empty list of numbers.",
     "def mean(nums): return sum(nums)/len(nums)"),
    ("Write `is_sorted(lst)` returning True if lst is non-decreasing.",
     "def is_sorted(lst): return all(lst[i]<=lst[i+1] for i in range(len(lst)-1))"),
    ("Write `binary_search(arr, target)` on a sorted list, returning index or -1.",
     "def binary_search(a,t):\n    l,r=0,len(a)-1\n    while l<=r:\n        m=(l+r)//2\n        if a[m]==t: return m\n        if a[m]<t: l=m+1\n        else: r=m-1\n    return -1"),
    ("Write `transpose(mat)` for a 2D list `mat`.",
     "def transpose(m): return [list(r) for r in zip(*m)]"),
    ("Write `roman_to_int(s)` converting a Roman numeral string to int.",
     "def roman_to_int(s):\n    v={'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}\n    t=0\n    for i,c in enumerate(s):\n        if i+1<len(s) and v[c]<v[s[i+1]]: t-=v[c]\n        else: t+=v[c]\n    return t"),
    ("Write `is_balanced(s)` for parentheses-balance checking on a string of `()[]{}`.",
     "def is_balanced(s):\n    st=[]; pairs={')':'(',']':'[','}':'{'}\n    for c in s:\n        if c in '([{': st.append(c)\n        elif c in ')]}':\n            if not st or st[-1]!=pairs[c]: return False\n            st.pop()\n    return not st"),
    ("Write `move_zeros(lst)` moving all zeros in a list to the end (preserve other order).",
     "def move_zeros(lst):\n    return [x for x in lst if x!=0]+[0]*lst.count(0)"),
    ("Write `longest_word(s)` returning the longest word in a sentence.",
     "def longest_word(s): return max(s.split(), key=len)"),
    ("Write `dict_invert(d)` returning a new dict with keys/values swapped.",
     "def dict_invert(d): return {v:k for k,v in d.items()}"),
    ("Write `triangle_area(a,b,c)` for sides a,b,c (Heron's formula).",
     "def triangle_area(a,b,c):\n    s=(a+b+c)/2\n    return (s*(s-a)*(s-b)*(s-c))**0.5"),
    ("Write `decimal_to_hex(n)` for non-negative integers as a string.",
     "def decimal_to_hex(n): return hex(n)[2:].upper() if n else '0'"),
]

SCIENCE_FALLBACK = [
    ("What gas do plants absorb during photosynthesis?", "Carbon dioxide (CO2)"),
    ("What is the unit of electric resistance?", "Ohm"),
    ("Which organelle generates most of a cell's ATP?", "Mitochondria"),
    ("State Newton's third law in one sentence.",
     "For every action there is an equal and opposite reaction."),
    ("What is the chemical symbol for gold?", "Au"),
    ("Which planet has the strongest gravitational pull in our solar system?", "Jupiter"),
    ("What pH value is neutral?", "7"),
    ("Name the process by which water vapor becomes liquid water.", "Condensation"),
    ("What is the speed of light in vacuum (approximate)?", "3×10^8 m/s"),
    ("Which blood type is the universal donor?", "O negative"),
    ("What molecule carries genetic information in most organisms?", "DNA"),
    ("Which subatomic particle has a positive charge?", "Proton"),
    ("What is the largest organ of the human body?", "Skin"),
    ("Define 'mole' (chemistry, briefly).",
     "A unit equal to ~6.022×10^23 particles (Avogadro's number)."),
    ("What gas makes up most of Earth's atmosphere?", "Nitrogen"),
    ("Which type of bond shares electrons?", "Covalent bond"),
    ("What instrument measures atmospheric pressure?", "Barometer"),
    ("Name the four DNA bases.", "Adenine, Thymine, Guanine, Cytosine"),
    ("In what part of the plant does photosynthesis primarily occur?",
     "Chloroplasts of the leaf"),
    ("What is the fundamental SI unit of mass?", "Kilogram (kg)"),
    ("Which law states that energy cannot be created or destroyed?",
     "First law of thermodynamics (conservation of energy)"),
    ("Which scientist proposed the theory of general relativity?", "Albert Einstein"),
    ("What does an ammeter measure?", "Electric current (in amperes)"),
    ("Which body system filters blood to produce urine?", "The urinary (renal) system"),
    ("What do enzymes do, in one sentence?",
     "They are biological catalysts that lower activation energy of reactions."),
    ("What is the boiling point of water at sea level (°C)?", "100"),
    ("Which gas is used by humans for cellular respiration?", "Oxygen"),
    ("Which subatomic particle has zero net charge?", "Neutron"),
    ("What process splits an atomic nucleus, releasing energy?", "Nuclear fission"),
    ("What is the function of red blood cells?",
     "Transport oxygen from lungs to body tissues."),
]

CHINESE_FALLBACK = [
    ("一个长方形的长是8米,宽是5米,它的周长是多少米?", "26"),
    ("速度40 km/h行驶3小时,行驶了多少千米?", "120"),
    ("一本书有240页,小明每天看20页,几天能看完?", "12"),
    ("水的化学式是什么?", "H2O"),
    ("光合作用的主要场所是?", "叶绿体"),
    ("万有引力定律的发现者是谁?", "牛顿"),
    ("化学反应中,什么不会改变?", "原子的种类和数量(质量守恒)"),
    ("声音在真空中能传播吗?", "不能"),
    ("地球公转一周大约需要多长时间?", "365天(一年)"),
    ("以下成语哪个意思是形容很有学问?A 学富五车 B 张牙舞爪", "A 学富五车"),
    ("古诗\"床前明月光\"的下一句是?", "疑是地上霜"),
    ("\"己所不欲,勿施于人\"出自哪部经典?", "《论语》"),
    ("\"千里之行,始于足下\"的作者(学派)是?", "老子(道家)"),
    ("圆周率π取小数点后两位是?", "3.14"),
    ("一元二次方程ax²+bx+c=0的求根公式是?",
     "x = (-b ± √(b²-4ac)) / (2a)"),
    ("我国第一部纪传体通史是?", "《史记》"),
    ("DNA的中文全称是?", "脱氧核糖核酸"),
    ("声波的传播需要什么?", "介质(气体、液体或固体)"),
    ("人体最大的器官是?", "皮肤"),
    ("酸碱中的pH值=7代表什么?", "中性"),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_seeds(
    subject: str,
    n_problems: int,
    *,
    use_huggingface: bool = False,
    seed: int = 0,
) -> list[tuple[str, str]]:
    """Return a list of (problem, ground_truth) for the given subject.

    use_huggingface: if True, attempts to load from real HF datasets
                     (math: MATH-500, code: MBPP, science: ScienceQA,
                     chinese: CMMLU). Falls back to the built-in seeds
                     on any failure.
    """
    rng = random.Random(seed)
    pool: list[tuple[str, str]]

    fallback_map = {
        "math": MATH_FALLBACK,
        "code": CODE_FALLBACK,
        "science": SCIENCE_FALLBACK,
        "chinese": CHINESE_FALLBACK,
    }
    if subject not in fallback_map:
        raise ValueError(f"Unknown subject '{subject}'. "
                         f"Choose from {list(fallback_map)}.")

    if use_huggingface:
        try:
            pool = _load_from_hf(subject)
            logger.info("Loaded %d seeds for '%s' from HuggingFace.",
                        len(pool), subject)
        except Exception as e:
            logger.warning("HuggingFace load failed for %s (%s); "
                           "using built-in fallback seeds.", subject, e)
            pool = list(fallback_map[subject])
    else:
        pool = list(fallback_map[subject])

    if n_problems > len(pool):
        logger.warning("Requested %d problems for '%s' but only %d available; "
                       "using all.", n_problems, subject, len(pool))
        n_problems = len(pool)

    rng.shuffle(pool)
    return pool[:n_problems]


def _load_from_hf(subject: str) -> list[tuple[str, str]]:
    """Fetch seeds from HuggingFace datasets. Requires `datasets` package."""
    from datasets import load_dataset  # lazy import

    if subject == "math":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        return [(r["problem"], str(r["answer"])) for r in ds]
    if subject == "code":
        ds = load_dataset("mbpp", split="test")
        return [(r["text"], r["code"]) for r in ds]
    if subject == "science":
        ds = load_dataset("derek-thomas/ScienceQA", split="test")
        return [(r["question"], str(r["answer"])) for r in ds if r.get("question")]
    if subject == "chinese":
        ds = load_dataset("haonan-li/cmmlu", "high_school_physics", split="test")
        return [(r["question"], str(r["answer"])) for r in ds]
    raise ValueError(subject)
