# 🚀 What is this project?

When building machine learning models, selecting the optimal combination of input features is one of the most important steps. A common approach is **Brute-Force Grid Search**, where every possible feature subset is evaluated.

However, this approach quickly becomes computationally infeasible because its time complexity grows exponentially:

\[
O(2^N)
\]

where **N** is the number of features.

For example:

- **10 features → 1,023 feature combinations**
- **20 features → 1,048,575 feature combinations**

Evaluating every combination on the entire dataset can take hours or even days.

---

# ⚡ Why is this engine faster?

This project introduces a custom **Successive Quartering Selection (SQS)** algorithm that dramatically reduces computation while maintaining high-quality feature selection.

Instead of exhaustively evaluating every feature subset on the full dataset, the algorithm progressively filters out poor-performing candidates through multiple rounds.

## 1. Data Budgeting

Rather than training every feature combination on **100% of the dataset**, the engine begins with only a **small fraction** (typically **15%**) of the data.

This allows hundreds or thousands of candidate feature subsets to be evaluated very quickly.

---

## 2. Aggressive Pruning (Tournament Style)

After each evaluation round, every feature subset is scored using a custom evaluation metric.

Only the **top 25%** of feature combinations survive.

The remaining **75%** are permanently discarded.

This tournament-style elimination drastically reduces the search space after every iteration.

---

## 3. Resource Scaling

Once weaker feature combinations have been eliminated, the surviving candidates are evaluated more rigorously.

For each new round:

- The training data budget is increased.
- More computational resources are allocated.
- Only the strongest candidates continue competing.

As the search progresses, fewer models remain, allowing deeper evaluation without excessive computation.

---

# 🛡️ The Mathematical Guard (MAE + Variance)

Aggressive pruning introduces a risk:

A genuinely strong feature subset may perform poorly on a small data sample simply due to randomness.

To reduce this risk, the engine uses a custom scoring function that considers both prediction accuracy **and** stability.

### Custom Score

\[
\text{Score} = \text{Mean MAE} + (1.96 \times \text{Variance})
\]

Where:

- **Mean MAE** measures prediction error.
- **Variance** measures consistency across cross-validation folds.
- **1.96** acts as a statistical confidence multiplier, penalizing unstable feature subsets.

This ensures that feature combinations are selected not only because they achieve low error, but also because they produce **consistent and reliable performance** across different validation splits.

---

# ✅ Key Advantages

- Much faster than exhaustive Brute-Force Grid Search
- Dramatically reduces computational cost
- Tournament-style elimination of weak feature subsets
- Progressive allocation of computational resources
- Stability-aware scoring using **MAE + Variance**
- Scales efficiently to high-dimensional datasets
