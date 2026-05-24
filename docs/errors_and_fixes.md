# Errors and Fixes

Real errors I hit while building this project, in the order they appeared. Each one has the exact message, what caused it, and what fixed it.

---

## 1. ModuleNotFoundError: No module named 'mlflow'

**When:** Running `python src/train.py` after installing requirements.

**Error:**

    ModuleNotFoundError: No module named 'mlflow'

**What happened:**
I had two Python versions on my machine - Python 3.10 and Python 3.11. Running `pip install -r requirements.txt` installed everything into Python 3.10. But `python` on my machine pointed to Python 3.11, which had nothing installed.

**Fix:**

    python -m pip install -r requirements.txt

Using `python -m pip` forces the pip that belongs to the same Python that runs the scripts. After this everything installed correctly.

**Lesson:** Always use `python -m pip` instead of just `pip` when you have multiple Python versions. They are not the same thing.

---

## 2. MLflow UI showed no experiments after running train.py

**When:** Opening `http://127.0.0.1:5000` after the training script ran.

**What happened:**
The MLflow UI was already open showing the Default experiment. I ran `python src/train.py` in the same terminal instead of a separate one. The runs were logged correctly but the UI was not refreshed and I was looking at the wrong experiment - Default instead of iris-random-forest.

**Fix:**
Open a second terminal and run the training script there while the UI terminal stays running. Then look in the left sidebar for the iris-random-forest experiment by name - it is separate from Default.

**Lesson:** MLflow creates a new experiment the first time it sees the name. It does not appear in the Default view. You have to click it in the sidebar.

---

## 3. All three runs showed accuracy=1.000 and f1=1.000

**When:** Comparing the three training runs in the MLflow UI.

**What happened:**
All three parameter combinations gave perfect scores. At first I thought something was wrong with the logging. But this is actually correct - Iris is a very clean, well-separated dataset and RandomForest handles it easily. All three parameter combinations were strong enough to classify it perfectly on the test set.

**What I did:**
Left the runs as they are and noted this in the README. A deliberately weak run will be added later to show what a failed threshold looks like in the pipeline. That is more useful than pretending the scores were different.

---

## 4. pytest could not import src.train

**When:** Running `python -m pytest tests/test_data_loading.py`

**Error:**

    ModuleNotFoundError: No module named 'src'

**What happened:**
pytest could not find the `src` folder because there was no `__init__.py` file in it. Python did not treat it as a package.

**Fix:**

    touch src/__init__.py
    touch tests/__init__.py

Adding empty `__init__.py` files to both folders tells Python to treat them as packages and the imports worked after that.

## 5. flake8 failed on the first CI run with multiple errors

**When:** After pushing the GitHub Actions workflow and watching the first run in the Actions tab.

**Errors:**
    F401 imported but unused
    F541 f-string is missing placeholders
    E302 expected 2 blank lines, found 1
    W292 no newline at end of file
    W293 blank line contains whitespace
    W291 trailing whitespace

**What happened:**
The code worked fine locally but the CI pipeline ran flake8 which caught several issues - unused imports left over from an earlier version of evaluate.py, f-strings written without any placeholders, inconsistent blank lines between functions, and missing newlines at the end of files. None of these broke the logic but flake8 treats them as errors and fails the pipeline.

**Fix:**
Fixed the unused imports and f-strings manually, then used autopep8 to clean up the whitespace and formatting issues automatically:

    autopep8 --in-place --aggressive src/evaluate.py src/register_model.py src/train.py tests/test_evaluate.py tests/test_data_loading.py

After that flake8 returned nothing and the pipeline went green.

**Lesson:** Running flake8 locally before pushing would have caught all of this. Added it as a habit going forward.

## 6. All training runs passed the threshold - no failure case to show

**When:** After merging the first PR and reviewing what the pipeline demonstrated.

**What happened:**
All three original runs scored F1 = 1.0 on Iris because RandomForest handles this dataset easily. The threshold gate existed in the code but was never actually triggered. A pipeline that only ever passes does not prove the gate works.

**Fix:**
Added a fourth parameter combination with n_estimators=2 and max_depth=1 - intentionally too weak for even a clean dataset. This run scores well below 0.85 and would be blocked from promotion. The three original runs still pass and the best one is still selected by evaluate.py.

**Lesson:**
Testing the failure case matters as much as testing the success case. This applies to ML pipelines the same way it applies to unit tests.

## 7. Wrong comparison operator broke the boundary test — caught by CI

**When:** While working on the ci-failure-demo branch.

**What happened:**
Changed `>=` to `>` in the `is_above_threshold` function in test_evaluate.py. This meant a model scoring exactly 0.85 would fail the threshold check when it should pass. The boundary test `test_model_exactly_at_threshold_passes` caught it immediately.

**Error in CI:**

    FAILED tests/test_evaluate.py::test_model_exactly_at_threshold_passes
    assert is_above_threshold(0.85) is True

**Fix:**
Changed `>` back to `>=`. One character difference but it changes the behaviour at the boundary case which is exactly the kind of thing that is easy to miss in a code review.

**Lesson:**
This is why the boundary test at exactly 0.85 was worth writing separately. A test for 0.95 would not have caught this. The CI pipeline stopped this from reaching main.

---