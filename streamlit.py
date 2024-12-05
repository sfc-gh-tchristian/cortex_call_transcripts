# in Packages, add snowflake-ml-python
import streamlit as st
import json
from snowflake.snowpark.context import get_active_session

st.title("Airline Transcript Analyser 	:airplane_departure:")
st.subheader("Powered by Snowflake Cortex :snowflake:")
session = get_active_session()

df_transcripts = session.table("TC_DB.GENAI.CUSTOMER_SUPPORT_TRANSCRIPTS")

df = df_transcripts.to_pandas()

call_selector = st.selectbox("Select a call", df["CALL_ID"].unique())

if call_selector:
    st.subheader(f"Call ID: {call_selector}")
    st.write("**Transcript**")
    transcript = df[df["CALL_ID"] == call_selector]["TRANSCRIPT"].values[0]
    st.caption(transcript)

    st.subheader("Generative AI Powered Analysis:")

    sql_cortex = f"""
        SELECT 
        SNOWFLAKE.CORTEX.SUMMARIZE(transcript) as SUMMARY,
        SNOWFLAKE.CORTEX.CLASSIFY_TEXT(transcript,['Complaint','Change a Booking','Place a Booking','Cancel a Booking','Request Information']) as TOPIC,
        SNOWFLAKE.CORTEX.SENTIMENT(transcript) as SENTIMENT,
        SNOWFLAKE.CORTEX.EXTRACT_ANSWER(transcript,'What was the booking reference?') as booking_ref,
        SNOWFLAKE.CORTEX.EXTRACT_ANSWER(transcript,'Where was the customer travelling to?') as location,
        SNOWFLAKE.CORTEX.COMPLETE('llama3.2-3b',CONCAT('What do you recommend as the next best action for this customer given the following transcript: ', transcript)) as NBA
        from TC_DB.GENAI.CUSTOMER_SUPPORT_TRANSCRIPTS 
        where call_id={call_selector};
    """

    if st.button(":sparkles: Cortex AI :sparkles:"):
        with st.spinner("Generating insight..."):
            
            with st.expander("SQL Query", expanded=False):
                st.code(sql_cortex, language="sql")
            
            with st.expander("Results", expanded=True):
                insight_df = session.sql(sql_cortex).to_pandas()

                st.subheader(f"Cortex Powered Insight for Call {call_selector}", divider=True)

                # Booking
                booking = json.loads(insight_df.at[0, 'BOOKING_REF'])
                b_answer = booking[0]['answer']
                b_score = booking[0]['score']
                st.write(f"*Booking Ref*: {b_answer} (Score: {b_score})")

                # Location
                location = json.loads(insight_df.at[0, 'LOCATION'])
                l_answer = location[0]['answer']
                l_score = location[0]['score']
                st.write(f"*Location*: {l_answer} (Score: {l_score})")

                st.write(f"*Topic*: {insight_df['TOPIC'][0]}")
                st.write(f"*Sentiment*: {insight_df['SENTIMENT'][0]}")
    
                st.divider()
                
                st.write("""***Summary of the call***:""")
                st.caption(f"""{insight_df['SUMMARY'][0]}""")
                
                st.divider()
                
                st.write("***Suggested next steps***:")
                st.caption(f"{insight_df['NBA'][0]}")
        

    st.divider()

    from snowflake.cortex import Complete
    
    st.write("**Custom Chat**")
    st.write('Use the input to build a custom prompt along with the transcript above.')
    
    model = st.selectbox("Which model would you like to use?",("mistral-large2", "llama3.1-405b", "llama3.1-70b", "llama3.1-8b", "reka-flash","snowflake-arctic","jamba-instruct", "mistral-7b", "llama3.2-3b", "llama3.2-1b"),placeholder="Select model...") 

    with st.expander("Python API Example", expanded=False):
        st.code("""from snowflake.cortex import Complete 
        reply = Complete(model, prompt)
        """, language="python")
    
    with st.form("prompt", clear_on_submit=False):
        text, btn = st.columns([6, 1])
        prompt = text.text_input("Enter prompt", placeholder="Where was the customer travelling to?", label_visibility="collapsed")
        submit = btn.form_submit_button("Submit", type="primary", use_container_width=True)

        system_p = f"""
        With the following transcript in mind consider the users question with that context. 
        <Transcript>
        {transcript}
        </Transcript>
        User question:
        """
        
        if submit and prompt:
            response = Complete(model, system_p+prompt).strip()
            st.write(response)


text = '''
import pandas as pd

# Sample DataFrame for demonstration
data = {'column_name': [[{"answer": "12345678xyz", "score": 0.11289931}]]}
df = pd.DataFrame(data)

# Extract the dictionary from the cell
cell_value = df.at[row_index, 'column_name']

# Extract the values for 'answer' and 'score'
answer = cell_value[0]['answer']
score = cell_value[0]['score']

print(f"Answer: {answer}")
print(f"Score: {score}")

'''
