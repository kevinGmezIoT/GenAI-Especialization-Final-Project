import os
from langchain_aws import ChatBedrock
from langchain_core.prompts import PromptTemplate
from langsmith import traceable

@traceable(name="Generate Customer Description")
def generate_inference_description(input_data: dict, model_id="amazon.titan-text-premier-v1:0"):
    """
    Generates a brief description of the customer using an LLM.
    """
    llm = ChatBedrock(model_id=model_id)

    template_text = """
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
    Edad: {Age}
    Sexo: {Sex}
    Trabajo: {Job}
    Alojamiento: {Housing}
    Cuentas de ahorro: {Saving_accounts}
    Cuenta corriente: {Checking_account}
    Monto del crédito: {Credit_amount}
    Duración: {Duration}
    Finalidad: {Purpose}
    
    Escribe tu respuesta a continuación:
    """
    
    # Map input keys to match template (handling aliases with spaces)
    variables = {
        "Age": input_data.get("Age"),
        "Sex": input_data.get("Sex"),
        "Job": input_data.get("Job"),
        "Housing": input_data.get("Housing"),
        "Saving_accounts": input_data.get("Saving accounts") or input_data.get("Saving accounts"),
        "Checking_account": input_data.get("Checking account") or input_data.get("Checking account"),
        "Credit_amount": input_data.get("Credit amount") or input_data.get("Credit amount"),
        "Duration": input_data.get("Duration"),
        "Purpose": input_data.get("Purpose")
    }
    
    prompt = PromptTemplate.from_template(template_text)
    chain = prompt | llm
    
    response = chain.invoke(variables)
    return response.content.strip()
