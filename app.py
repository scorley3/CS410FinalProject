from flask import Flask, render_template, redirect, url_for, request
from flask import Blueprint, render_template, send_from_directory


app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")