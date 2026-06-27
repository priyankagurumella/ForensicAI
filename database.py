import sqlite3

conn = sqlite3.connect("news.db")

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS prediction_history(
id INTEGER PRIMARY KEY AUTOINCREMENT,
news TEXT,
prediction TEXT,
confidence REAL
)
""")

conn.commit()

conn.close()

print("Database Created Successfully")