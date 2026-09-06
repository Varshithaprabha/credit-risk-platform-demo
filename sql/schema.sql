-- Reference schema for the SQLite database built by src/data/loader.py.
-- Tables are actually created dynamically (via pandas.to_sql) from whichever
-- Home Credit CSVs are present in ./data, so this file is documentation of
-- the expected shape rather than a script you need to run by hand.

-- application: one row per loan application (main modeling + chatbot table)
--   SK_ID_CURR        INTEGER PRIMARY KEY  -- application id
--   TARGET            INTEGER              -- 1 = defaulted, 0 = repaid (train only)
--   NAME_CONTRACT_TYPE, CODE_GENDER, AMT_INCOME_TOTAL, AMT_CREDIT, AMT_ANNUITY,
--   NAME_EDUCATION_TYPE, NAME_FAMILY_STATUS, NAME_HOUSING_TYPE, OCCUPATION_TYPE,
--   DAYS_BIRTH, DAYS_EMPLOYED, EXT_SOURCE_1/2/3, ... (120+ columns total)

-- bureau: applicant's credit history at other financial institutions
--   SK_ID_CURR (FK -> application), SK_ID_BUREAU, CREDIT_ACTIVE, AMT_CREDIT_SUM, ...

-- bureau_balance: monthly balance snapshots for each bureau credit
--   SK_ID_BUREAU (FK -> bureau), MONTHS_BALANCE, STATUS

-- previous_application: applicant's previous loan applications with Home Credit
--   SK_ID_PREV, SK_ID_CURR (FK -> application), NAME_CONTRACT_STATUS, ...

-- pos_cash_balance / credit_card_balance / installments_payments:
--   monthly behavioural data on previous POS/cash loans and credit cards,
--   keyed by SK_ID_PREV / SK_ID_CURR

CREATE INDEX IF NOT EXISTS idx_app_id ON application(SK_ID_CURR);
CREATE INDEX IF NOT EXISTS idx_bureau_id ON bureau(SK_ID_CURR);
CREATE INDEX IF NOT EXISTS idx_prev_id ON previous_application(SK_ID_CURR);
