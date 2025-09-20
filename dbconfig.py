from sqlalchemy import create_engine, text

db_username = "postgres"
db_pass = "ebackender44"
db_name = "student_dataset_analyze"

engine = create_engine(f"postgresql://{db_username}:{db_pass}@localhost/{db_name}")

# Ping database with SELECT 1 query to test if it's work
"""
with engine.connect() as conn:
  result = conn.execute(text("SELECT 1"))
  print(result.scalar())
"""

