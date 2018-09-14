from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, FloatField
from wtforms.validators import DataRequired,Email,EqualTo



class ClassifyForm(FlaskForm):

      submit = SubmitField('Classify')


class AIForm(FlaskForm):
      ai_info = StringField('Username', validators=[DataRequired()])

      submit = SubmitField('Close')