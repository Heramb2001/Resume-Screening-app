import streamlit as st
import pickle
import re
import nltk

nltk.download('punkt')
nltk.download('stopwords')

clf = pickle.load(open('clf.pkl','rb'))
tfidfd = pickle.load(open('tfidf.pkl','rb'))

def clean_resume(resume_text):
    clean_text = re.sub('http\S+\s*', ' ', resume_text)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+', '', clean_text)
    clean_text = re.sub('@\S+', '  ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return clean_text
# web app
def main():
    st.markdown("<h2 style='color: #007BFF;'>Resume Analyser:</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader('Upload Resume below...', type=['txt','pdf'])

    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            
            resume_text = resume_bytes.decode('latin-1')

        cleaned_resume = clean_resume(resume_text)
        input_features = tfidfd.transform([cleaned_resume])
        prediction_id = clf.predict(input_features)[0]
        
       
        st.markdown("<h2 style='color: #1abc9c;font-size: 30px'>Predicted Category:</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 24px; font-weight: bold; color: green;'>{prediction_id}</p>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()