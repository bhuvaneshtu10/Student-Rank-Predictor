import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import joblib

# 1) Locate dataset (prefers unbiased, falls back to alt file)
candidates = [
    "Students-Performance-Dataset.csv",
    "Students_Grading_Dataset_Biased.csv",
]
csv_path = next((p for p in candidates if os.path.exists(p)), None)
if csv_path is None:
    raise FileNotFoundError(f"Place one of these files in the working folder: {candidates}")

print(f"Using dataset: {csv_path}")

# 2) Load
df_raw = pd.read_csv(csv_path)

# 3) Standardize and robust alias mapping
def norm(s):
    # Aggressively remove spaces, underscores, make consistent
    return s.strip().replace("  ", " ").replace(" ", "").replace("_", "").lower()

df = df_raw.copy()
original_columns = df.columns.tolist()
df.columns = [norm(c) for c in df.columns]

# Key columns from metadata, mapped to normalized names
expected = {
    "studentid": "StudentID", "firstname": "FirstName", "lastname": "LastName", "email": "Email",
    "gender": "Gender", "age": "Age", "department": "Department", "attendance": "Attendance",
    "midtermscore": "MidtermScore", "finalscore": "FinalScore", "assignmentsavg": "AssignmentsAvg",
    "quizzesavg": "QuizzesAvg", "participationscore": "ParticipationScore", "projectsscore": "ProjectsScore",
    "totalscore": "TotalScore", "grade": "Grade", "studyhoursperweek": "StudyHoursperWeek",
    "extracurricularactivities": "ExtracurricularActivities", "internetaccessathome": "InternetAccessatHome",
    "parenteducationlevel": "ParentEducationLevel", "familyincomelevel": "FamilyIncomeLevel",
    "stresslevel1-10": "StressLevel1-10", "sleephourspernight": "SleepHoursperNight"
}

# Rebuild DataFrame with only expected columns (if present), keep original column names for output
norm_to_original = {norm(name): name for name in original_columns}
cols_found = [norm_to_original.get(k, None) for k in expected.keys() if k in norm_to_original]
df_use = df_raw[cols_found] if cols_found else df_raw.copy()
df_use.columns = [expected[k] for k in expected.keys() if k in norm_to_original]

print("Columns after cleaning/aliasing:", df_use.columns.tolist())

# 4) Coerce numeric columns
numeric_cols = [
    "Age","Attendance","MidtermScore","FinalScore","AssignmentsAvg","QuizzesAvg",
    "ParticipationScore","ProjectsScore","TotalScore","StudyHoursperWeek",
    "SleepHoursperNight"
]
for col in numeric_cols:
    if col in df_use.columns:
        df_use[col] = pd.to_numeric(df_use[col], errors="coerce")

# 5) Clean Yes/No and fill categorical
for col in ["ExtracurricularActivities", "InternetAccessatHome"]:
    if col in df_use.columns:
        df_use[col] = (
            df_use[col]
            .astype(str)
            .str.lower()
            .str.strip()
            .replace(
                {'yes':'Yes','no':'No','y':'Yes','n':'No','1':'Yes','0':'No','nan':'No'}
            )
            .fillna("No")
        )

# 6) REGRESSION - TotalScore
if "TotalScore" not in df_use.columns:
    print("ERROR: 'TotalScore' column missing. These are present:", df_use.columns.tolist())
else:
    reg_df = df_use.dropna(subset=["TotalScore"])
    print(f"Rows available for regression: {len(reg_df)}")
    if len(reg_df) >= 50:
        drop_cols = [c for c in ["StudentID", "FirstName", "LastName", "Email", "TotalScore", "Grade"] if c in reg_df.columns]
        X_reg = reg_df.drop(columns=drop_cols, errors="ignore")
        y_reg = reg_df["TotalScore"]

        cat_cols = [c for c in X_reg.columns if X_reg[c].dtype == "object"]
        num_cols = [c for c in X_reg.columns if c not in cat_cols]
        reg_pre = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ],
            remainder="drop"
        )
        reg_model = RandomForestRegressor(
            n_estimators=350,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        )
        reg_pipe = Pipeline(steps=[("pre", reg_pre), ("model", reg_model)])

        Xtr, Xte, ytr, yte = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
        reg_pipe.fit(Xtr, ytr)
        yp = reg_pipe.predict(Xte)
        mse = mean_squared_error(yte, yp)
        r2 = r2_score(yte, yp)
        print(f"[Regression] TotalScore -> MSE: {mse:.3f} | RMSE: {mse**0.5:.3f} | R2: {r2:.3f}")
        joblib.dump(reg_pipe, "total_score_regressor.pkl")
        print("Saved: total_score_regressor.pkl")
    else:
        print("[Regression] Skipped: insufficient rows after cleaning.")

# 7) CLASSIFICATION - Grade
if "Grade" not in df_use.columns:
    print("[Classification] Skipped: 'Grade' not present. Columns:", df_use.columns.tolist())
else:
    clf_df = df_use.dropna(subset=["Grade"]).copy()
    clf_df = clf_df[clf_df["Grade"].astype(str).str.upper().isin(list("ABCDF"))]
    print(f"Rows available for classification: {len(clf_df)}")
    if len(clf_df) >= 50:
        drop_cols = [c for c in ["StudentID", "FirstName", "LastName", "Email", "Grade", "TotalScore"] if c in clf_df.columns]
        X_clf = clf_df.drop(columns=drop_cols, errors="ignore")
        y_clf = clf_df["Grade"].astype(str).str.upper()

        cat_cols_c = [c for c in X_clf.columns if X_clf[c].dtype == "object"]
        num_cols_c = [c for c in X_clf.columns if c not in cat_cols_c]
        clf_pre = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(with_mean=False), num_cols_c),
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols_c),
            ],
            remainder="drop"
        )
        clf_model = RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        clf_pipe = Pipeline(steps=[("pre", clf_pre), ("model", clf_model)])
        Xtr_c, Xte_c, ytr_c, yte_c = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf)
        clf_pipe.fit(Xtr_c, ytr_c)
        yp_c = clf_pipe.predict(Xte_c)
        acc = accuracy_score(yte_c, yp_c)
        print(f"[Classification] Grade -> Accuracy: {acc:.3f}")
        print(classification_report(yte_c, yp_c))
        joblib.dump(clf_pipe, "grade_classifier.pkl")
        print("Saved: grade_classifier.pkl")
    else:
        print("[Classification] Skipped: insufficient labeled rows.")

print("---- Script complete ----")
