import smtplib
from email.message import EmailMessage
import smtplib, ssl
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import settings

def send_an_email(subject='Hi Senjing!', info='This is an auto-generated info from python.'):

	# print('I am trying to send an email via python')

	msg = EmailMessage()
	msg.set_content(info)

	me = settings.EMAIL
	you = ['jerryzsj@icloud.com']

	msg['Subject'] = f'Important news from Alis: '+subject
	msg['From'] = me
	msg['To'] = you

	# Send the message via our own SMTP server.
	# s = smtplib.SMTP('localhost')
	# s.send_message(msg)
	# s.quit()

	port = 587  # For SSL
	password = settings.PASSWORD

	# Create a secure SSL context
	context = ssl.create_default_context()  

	server = smtplib.SMTP("smtp-mail.outlook.com",port)
	server.starttls()
	server.login(me, password)
	# server.ehlo()
	server.send_message(msg)
	# server.ehlo()
	server.quit()


if __name__=='__main__':
	send_an_email(subject='Colab finished the job!',info='This is an automatic info from python. ')
