"""Local, transactional evidence journal. Real sessions never modify v2.1."""
import json
import sqlite3
from pathlib import Path

class Store:
    def __init__(self,path):
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(str(path),check_same_thread=False)
        self.db.row_factory=sqlite3.Row
        self.db.executescript('''
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS profiles(id TEXT PRIMARY KEY,name TEXT NOT NULL,created REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS interactions(id TEXT PRIMARY KEY,learner TEXT NOT NULL,skill TEXT NOT NULL,task TEXT NOT NULL,payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY AUTOINCREMENT,learner TEXT NOT NULL,skill TEXT NOT NULL,interaction TEXT NOT NULL,at REAL NOT NULL,kind TEXT NOT NULL,payload TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS event_stream ON events(learner,skill,id);
        CREATE TABLE IF NOT EXISTS requests(id TEXT PRIMARY KEY,result TEXT NOT NULL);
        ''')
        self.db.commit()

    def events(self,learner,skill=None):
        query='SELECT * FROM events WHERE learner=?'; args=[learner]
        if skill: query+=' AND skill=?'; args.append(skill)
        return [dict(json.loads(r['payload']),id=r['id'],at=r['at'],kind=r['kind'],skill=r['skill'],interaction=r['interaction']) for r in self.db.execute(query+' ORDER BY id',args)]

    def interactions(self,learner):
        return [json.loads(r[0]) for r in self.db.execute('SELECT payload FROM interactions WHERE learner=? ORDER BY rowid',(learner,))]

    def save(self,item):
        self.db.execute('INSERT INTO interactions VALUES(?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload',
                        (item['id'],item['learner'],item['skill'],item['task'],json.dumps(item)))
