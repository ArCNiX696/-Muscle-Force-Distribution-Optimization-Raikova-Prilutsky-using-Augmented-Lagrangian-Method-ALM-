# 💪🧠 Muscle Force Distribution Optimization using Augmented Lagrangian Method (ALM)

---

## 📄 Project Description

This project contains two numerical optimization exercises related to muscle force distribution. Both exercises were solved using gradient-based methods and include analytical derivation of gradients and constraints.

---

## 🧩 Exercise 1: Quadratic Function with Equality Constraint (DFP Method)

- **Objective**: Minimize a quadratic objective function  
- **Constraint**: One linear equality constraint  
- **Method used**: Davidon–Fletcher–Powell (DFP)  
- **Penalty approach**: Reformulated using squared constraint penalization  
- **Line Search**: Armijo backtracking for step-size selection  
- **Final Solution**:  
  - `x ≈ [0.857, 2.571, 1.714]`  
  - `F(x) ≈ 10.28`  
- 📈 The algorithm converged in **3 iterations**, and you can find the convergence plot in the output.

---

## 📌 Exercise 2: Muscle Force Distribution Problem (Based on Raikova & Prilutsky)

- **Paper Reference**:  
  **"Sensitivity of predicted muscle forces to parameters of the optimization-based human leg model revealed by analytical and numerical analyses"**  
  *Raikova, R.T. & Prilutsky, B.I., Journal of Biomechanics, 2001*  
  DOI: [10.1016/S0021-9290(01)00097-5](https://doi.org/10.1016/S0021-9290(01)00097-5)  
  📄 *(The PDF of the paper is included in the repository)*

- **Method used**:  
  - Augmented Lagrangian Method (ALM)  
  - Gradient Descent (GD) for iterative optimization  
  - Analytical derivation of constraint gradients and Jacobian  

- **Constraints**:  
  - 3 linear equality constraints  
  - Non-negativity of all forces (handled via projection)

- **Final Results**:
  - Optimal Forces: `F1 to F9` listed in terminal output
  - Objective Value: `Z(F) ≈ 626.27`

---

## 📊 Report and Analysis

> ✍️ A detailed analysis and explanation of the derivations, methodology, convergence plots, and results for both exercises is provided in:  
> 👉 **[Report.pdf](Report.pdf)**

---

## 📁 Repository Structure

- `exercise1_dfp.py`: Implementation of DFP method for Exercise 1  
- `exercise2_alm.py`: ALM + GD for muscle force optimization  
- `Report.pdf`: Final project report  
- `Raikova_Priltusky_Paper.pdf`: Original paper for reference

---

## 🧪 Visualization

### 📈 Objective vs Penalized Function (ALM)
![Objective vs Penalized](assets/ALM_convergence.png)

### 🔻 Constraint Violation (Log Scale)
![Constraint Violation](assets/Constraint_violation_log.png)

---

## 📥 Cloning the Repository

```bash
git clone https://github.com/ArCNiX696/-Muscle-Force-Distribution-Optimization-Raikova-Prilutsky-using-Augmented-Lagrangian-Method-ALM-.git
cd -Muscle-Force-Distribution-Optimization-Raikova-Prilutsky-using-Augmented-Lagrangian-Method-ALM-

