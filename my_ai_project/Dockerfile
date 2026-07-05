FROM python:3.9
WORKDIR /app
COPY . .
RUN pip install flask flask-cors opencv-python-headless pandas tensorflow
RUN pip install gunicorn
EXPOSE 7860
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "web_app.app:app"]