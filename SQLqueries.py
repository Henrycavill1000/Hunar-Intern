import sqlite3

conn = sqlite3.connect(':memory:')
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE Student (
    RollNo INTEGER PRIMARY KEY,
    Name TEXT NOT NULL,
    Marks INTEGER,
    Aadhar TEXT UNIQUE,
    Address TEXT
)
''')

cursor.execute('''
CREATE TABLE Course (
    RollNo INTEGER,
    Course TEXT,
    Course_Duration TEXT,
    FOREIGN KEY(RollNo) REFERENCES Student(RollNo)
)
''')

students = [
    (1, 'Rohan', 25, '1234', 'Delhi'),
    (2, 'Anjali', 85, '2345', 'Mumbai'),
    (3, 'Ritika', 65, '3456', 'Pune'),
    (4, 'Rahul', 29, '4567', 'Noida'),
    (5, 'Maya', 70, '5678', 'Chennai'),
]

courses = [
    (1, 'BCA', '3 years'),
    (2, 'BBA', '3 years'),
    (3, 'BCA', '3 years'),
    (4, 'MCA', '2 years'),
    (5, 'BCA', '3 years'),
]

cursor.executemany("INSERT INTO Student VALUES (?, ?, ?, ?, ?)", students)
cursor.executemany("INSERT INTO Course VALUES (?, ?, ?)", courses)

print("\n1. Average of Marks:")
cursor.execute("SELECT AVG(Marks) FROM Student")
print("Average Marks:", cursor.fetchone()[0])

print("\n2. Names in Ascending Order:")
cursor.execute("SELECT Name FROM Student ORDER BY Name ASC")
for row in cursor.fetchall():
    print(row[0])

print("\n3. RollNo and Names of Students scoring below 30:")
cursor.execute("SELECT RollNo, Name FROM Student WHERE Marks < 30")
for row in cursor.fetchall():
    print(row)

print("\n4. RollNo of Students whose names start with 'R':")
cursor.execute("SELECT RollNo FROM Student WHERE Name LIKE 'R%'")
for row in cursor.fetchall():
    print(row[0])

print("\n5. RollNo of Students pursuing BCA:")
cursor.execute("SELECT RollNo FROM Course WHERE Course = 'BCA'")
for row in cursor.fetchall():
    print(row[0])

conn.close()