import pandas as pd
from google import genai
import os
from dotenv import load_dotenv

# Load the secure API key from the .env file
load_dotenv()
client = genai.Client()

# 1. Load the database ONCE
DB_PATH = 'data/processed_data/smd_database.csv'
if os.path.exists(DB_PATH):
    smd_df = pd.read_csv(DB_PATH)
    smd_df['Code'] = smd_df['Code'].astype(str) 
else:
    smd_df = None
    print(f"Warning: {DB_PATH} not found!")

# --- 2. CHAT SESSION STORAGE ---
chat_sessions = {}

def lookup_component(smd_code):
    if smd_df is None: return None
    result = smd_df[smd_df['Code'] == str(smd_code)]
    if not result.empty:
        return result.iloc[0].to_dict()
    return None

def get_doctor_advice(defect_type, component_info, board_context="assembled", user_message=""):
    
    is_general = defect_type.lower() in ["general inquiry", "none", ""]
    
    # Wrap the ENTIRE process in a try-except block so rate limits don't crash the server
    try:
        if is_general:
            session_key = "general_tutor_session"
            
            if session_key not in chat_sessions:
                chat_sessions[session_key] = client.chats.create(model='gemini-2.0-flash')
                system_instr = """
                You are a highly knowledgeable and friendly Electronics and Communication Engineering (ECE) Tutor.
                The user will ask you general questions about PCBs, electronic components, physics, and circuit design.
                
                CRITICAL INSTRUCTIONS:
                1. Act like a helpful AI assistant. Answer their questions directly, clearly, and accurately.
                2. DO NOT mention camera scans, AOI systems, "General Inquiry" anomalies, or Action Plans. 
                3. Do NOT act like you are looking at a broken board. Just answer the user's question.
                4. Do NOT use markdown asterisks (**) for bolding. Use plain text or standard capitalization.
                """
                chat_sessions[session_key].send_message(system_instr)
                
        else:
            session_key = f"repair_session_{board_context}_{defect_type}"
            
            if session_key not in chat_sessions:
                chat_sessions[session_key] = client.chats.create(model='gemini-2.0-flash')
                
                if board_context == "bare": role = "Expert PCB Fabrication Inspector"
                elif board_context == "track": role = "Master Trace Repair Technician"
                else: role = "Senior SMT Assembly Engineer"
                
                system_instr = f"""
                You are the '{role}'. A scanner has detected a real defect: '{defect_type}'.
                Context: {board_context} board. Component: {component_info.get('Device', 'General area')}.
                
                INSTRUCTIONS:
                - Provide a professional repair 'Action Plan' (Root Cause, Repair Steps, Verification).
                - Keep the tone highly technical and strict.
                - Answer follow-up questions specifically about fixing this '{defect_type}'.
                - Do NOT use markdown asterisks (**) for bolding.
                """
                chat_sessions[session_key].send_message(system_instr)

        # Send the user's message
        response = chat_sessions[session_key].send_message(user_message)
        return response.text.replace("**", "")

    except Exception as e:
        error_msg = str(e)
        # Handle the specific Quota/Rate Limit error gracefully
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            return "API Speed Limit Reached: Please wait 15 to 30 seconds and try asking again!"
        
        return f"Communication Error: {error_msg}"