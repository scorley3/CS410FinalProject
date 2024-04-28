from flask import Flask, render_template, redirect, url_for, request
from flask import Blueprint, render_template, send_from_directory
from home.home import home_blueprint
import os 

os.environ['FLASK_DEBUG'] = 'True'

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

app.register_blueprint(home_blueprint)

if __name__ == "__main__":
    app.run(debug=True)