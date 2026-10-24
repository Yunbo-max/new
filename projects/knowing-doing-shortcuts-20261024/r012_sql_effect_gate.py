"""R012 compact source companion. The complete 14-test SQLite implementation
is in R012_research_checkpoint.zip. No LLM inference is performed here."""
import sqlite3

def world():
    db=sqlite3.connect(":memory:")
    db.execute("create table orders(id text primary key, amount integer, status text)")
    db.executemany("insert into orders values(?,?,?)",
                   [("A",10,"pending"),("B",20,"pending")])
    return db

def snapshot(db):
    return db.execute("select id,amount,status from orders order by id").fetchall()

def act(db,mode):
    before=snapshot(db)
    if mode=="normal":
        db.execute("update orders set status='refunded' where id='A'")
    elif mode=="wrong_target":
        db.execute("update orders set status='refunded' where id='B'")
    elif mode=="silent_noop":
        pass
    else:raise ValueError(mode)
    # These untrusted receipts all claim success.
    return {"receipt":{"success":True},"before":before,"after":snapshot(db)}

def check(trace,prior_evidence):
    # prior_evidence is a list of (time, entity, attribute, value).
    proof={(entity,field,value) for time,entity,field,value in prior_evidence if time<10}
    if ("A","amount",10) not in proof or ("A","status","pending") not in proof:
        return False
    old={r[0]:r for r in trace["before"]}
    new={r[0]:r for r in trace["after"]}
    return (old["A"]==("A",10,"pending") and
            new["A"]==("A",10,"refunded") and
            all(new[x]==old[x] for x in old if x!="A"))

if __name__=="__main__":
    evidence=[(1,"A","amount",10),(2,"A","status","pending")]
    for name,expected in (("normal",True),("silent_noop",False),
                          ("wrong_target",False)):
        db=world(); r=act(db,name)
        assert r["receipt"]["success"] is True
        assert check(r,evidence)==expected
    print("3 / 3 deterministic SQL state-effect checks pass; no LLM run.")
