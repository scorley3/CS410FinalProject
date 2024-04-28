from flask import Blueprint, render_template, redirect, url_for, request, session

home_blueprint = Blueprint('home', __name__)

@home_blueprint.route('/')
def home():
    return render_template('home.html', songs=[])

@home_blueprint.route('/generate', methods=['GET'])
def load(): 
    # code to generate songs to display 
    return render_template('index.html')