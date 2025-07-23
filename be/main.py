from fastapi import fastapi

app = fastapi()

@app.get("/")
def root():
    print("got msg")
    return {"Message":"Hello World"}
