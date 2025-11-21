import pandas as pd
import time
import os
from langchain_aws import ChatBedrock
from langchain_core.prompts import PromptTemplate

def add_description(data, model_id="amazon.titan-text-premier-v1:0", profile_name="bedrock-user-admin"):
    print("Generating descriptions...")
    try:
        llm = ChatBedrock(
            credentials_profile_name=profile_name,
            model_id=model_id
        )
    except Exception as e:
        print(f"Error initializing Bedrock with profile {profile_name}: {e}")
        print("Attempting with default credentials...")
        llm = ChatBedrock(model_id=model_id)

    text = """
    Eres un experto en riesgos crediticio bancario.
    Se te proveerá una serie de datos descritos a continuación:
    Edad: Edad de la persona
    Sexo: Sexo de la persona
    Trabajo: ( 0 - unskilled and non-resident, 1 - unskilled and resident, 2 - skilled, 3 - highly skilled)
    Alojamiento: Tipo de alojamiento
    Cuentas de ahorro: Tipo de cuenta de ahorro
    Cuenta corriente: Tipo de cuenta corriente
    Monto del crédito: Monto de crédito
    Duración (meses): Tiempo de préstamo
    Finalidad: Motivo del préstamo
    
    Tu tarea es describir los datos presentados en un máximo de 30 palabras con su relación con el riesgo crediticio.
    
    Estos son los datos:
    Edad: {age}
    Sexo: {sex}
    Trabajo: {job}
    Alojamiento: {Housing}
    Cuentas de ahorro: {Saving_accounts}
    Cuenta corriente: {Checking_account}
    Monto del crédito: {Credit_amount}
    Duración: {Duration}
    Finalidad: {Purpose}
    
    Escribe tu respuesta a continuación:
    """
    
    prompt_template = PromptTemplate.from_template(text)
    results = []
    
    # Processing in batches or loops. For simplicity/robustness in script:
    for index, row in data.iterrows():
        try:
            result = prompt_template.invoke({
                "age": row['Age'],
                "sex": row['Sex'],
                "job": row['Job'],
                "Housing": row['Housing'],
                "Saving_accounts": row['Saving accounts'],
                "Checking_account": row['Checking account'],
                "Credit_amount": row['Credit amount'],
                "Duration": row['Duration'],
                "Purpose": row['Purpose']
            })
            classification = llm.invoke(input=result)
            results.append(classification.content)
            # Rate limiting/Sleep to avoid throttling if necessary, though Titan is usually fast.
            # time.sleep(0.5) 
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            results.append("Error generating description")

    data['description'] = results
    return data

def add_target(data, model_id="anthropic.claude-3-5-sonnet-20240620-v1:0", profile_name="bedrock-user-admin"):
    print("Generating targets...")
    try:
        llm = ChatBedrock(
            credentials_profile_name=profile_name,
            model_id=model_id
        )
    except Exception as e:
        print(f"Error initializing Bedrock with profile {profile_name}: {e}")
        print("Attempting with default credentials...")
        llm = ChatBedrock(model_id=model_id)
    
    text = """
    You are an expert in bank credit risk.
    Your task is to classify the credit risk as 'good risk' or 'bad risk'.
    
    Examples:
    Description: A 67 year old man requested a loan of 1169 for a TV, he is skilled, has little savings, his own home and has requested a loan for 6 months.
    Answer: bad risk
    
    Description: A 22 year old woman requested a loan of 5951 euros for 48 months to buy a radio or television. She is a skilled worker and owns her own home. She has a small savings account and a moderate current account.
    Answer: good risk
    
    The following is the description you must classify and is important your answer should be only 'good risk' or 'bad risk':
    
    Description: {description}
    Answer:
    """
    
    prompt_template = PromptTemplate.from_template(text)
    
    results = []
    for index, row in data.iterrows():
        try:
            description = row['description']
            result = prompt_template.invoke({"description": description})
            classification = llm.invoke(input=result)
            results.append(classification.content.strip())
            # time.sleep(0.5)
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            results.append("Error generating target")
    
    data['target'] = results
    return data

if __name__ == "__main__":
    # Define paths
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_path = os.path.join(base_path, "data", "raw", "credit_risk_reto.csv")
    output_path = os.path.join(base_path, "data", "processed", "credit_risk_enriched.csv")
    
    # Create processed directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if os.path.exists(input_path):
        print(f"Loading data from {input_path}")
        df = pd.read_csv(input_path)
        
        # Basic cleaning before enrichment if needed, or just pass raw
        # The notebook did imputation before description, let's do basic imputation here or in preprocess?
        # The notebook `clean_data` did imputation. Let's replicate basic imputation here to ensure Bedrock gets clean data.
        
        categorical_cols = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
        # Simple fill for text generation purposes
        for col in categorical_cols:
            df[col] = df[col].fillna('unknown')
            
        df = add_description(df)
        df = add_target(df)
        
        print(f"Saving enriched data to {output_path}")
        df.to_csv(output_path, index=False)
    else:
        print(f"Input file not found at {input_path}")
