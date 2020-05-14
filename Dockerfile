FROM python:3.8.2-buster

# Install dependencies
RUN pip3 install pandas
RUN pip3 install sklearn
RUN pip3 install flask
RUN pip3 install flask-restplus
RUN pip3 install fastai
RUN pip3 install spacy
RUN pip3 install Werkzeug==0.16.1
# Create working directory and copy code
RUN mkdir /stock_sentiment_ulmfit
WORKDIR /stock_sentiment_ulmfit
COPY api.py /stock_sentiment_ulmfit/api.py
COPY stock_sentiment_model.pkl /stock_sentiment_ulmfit/stock_sentiment_model.pkl
# Run command
CMD python3 api.py