import os 
from fastapi import FastAPI
from pydantic import BaseModel,Field
from typing import Optional
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

load_dotenv()

INSTAGRAM_PROMPT = """
You are a professional social media copywriter specialized in Indian handmade and heritage products.

TASK:
Generate an Instagram caption for a handmade artisan product.

STRICT CONSTRAINTS (MANDATORY):
- Maximum 2 lines
- Do NOT mention price
- Do NOT include hashtags
- Tone must be emotional and aesthetic
- Focus on craftsmanship, tradition, and authenticity
- Language must feel natural, not promotional
- No call-to-action like "Buy now", "Order", or "DM"

PRODUCT DETAILS:
Product Name: {product_name}
Material: {material}
Craft Technique: {craft_type}
Region / Origin: {region}
Artisan Explanation: {artisan_story}

OUTPUT FORMAT:
Return only the caption text.
No headings.
No extra commentary.
"""
WHATSAPP_PROMPT = """
        You generate WhatsApp broadcast messages for artisan products.

        CONSTRAINTS:
        - Price must be mentioned
        - Clear and direct tone
        - 3–4 short lines
        - No emotional storytelling
        - Simple buying intent
        - No hashtags

        PRODUCT DETAILS:
        Product: {product_name}
        Material: {material}
        Craft: {craft_type}
        Region: {region}
        Price: ₹{price}

        OUTPUT:
        Return only the WhatsApp message text.
"""

WEBSITE_PROMPT = """
        You write detailed website product descriptions for handmade products.

        CONSTRAINTS:
        - Minimum 150 words
        - Informative and structured
        - Explain material, process, and usage
        - Preserve cultural context
        - SEO-friendly language
        - No emojis, no CTA phrases

        PRODUCT DETAILS:
        Product Name: {product_name}
        Material: {material}
        Craft Technique: {craft_type}
        Region: {region}
        Artisan Explanation: {artisan_story}

        OUTPUT FORMAT:
        Return only the description text.
"""
AD_PROMPT = """
        You create high-conversion ad copy.

        CONSTRAINTS:
        - Maximum 15 words
        - One clear benefit
        - Strong call-to-action mandatory
        - No background story

        PRODUCT DETAILS:
        Product: {product_name}
        Key Benefit: {primary_benefit}

        OUTPUT:
        Return only the ad copy text.
"""


instagram_prompt = PromptTemplate(
    template= INSTAGRAM_PROMPT,
    input_variables = [
    "product_name",
    "material",
    "craft_type",
    "region",
    "artisan_story"
]

)

whatsapp_prompt = PromptTemplate(
    template= WHATSAPP_PROMPT,
    input_variables = [
    "product_name",
    "material",
    "craft_type",
    "region",
    "artisan_story",
    "price"
]

)

website_prompt = PromptTemplate(
    template= WEBSITE_PROMPT,
    input_variables=[
        "product_name",
        "material",
        "craft_type",
        "region",
        "artisan_story"
    ]
)

ad_prompt = PromptTemplate(
    template= AD_PROMPT,
    input_variables=[
        "product_name",
        "primary_benefit"
    ]
)


model = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.8
    
)
parser = StrOutputParser()


instagram_chain = instagram_prompt | model | parser
whatsapp_chain  = whatsapp_prompt  | model | parser
website_chain   = website_prompt   | model | parser
ad_chain        = ad_prompt        | model | parser


data = {
    "product_name": "Handcrafted Terracotta Diya Set",
    "material": "Natural red clay",
    "craft_type": "Traditional hand-thrown pottery",
    "region": "Molela, Rajasthan",
    "artisan_story": "Each diya is shaped by hand on a traditional wheel, sun-dried, and kiln-fired using age-old terracotta techniques passed down through generations.",
    "price": 499,
    "primary_benefit": "Authentic handmade festive decor"
}
 
output = {
    "instagram": instagram_chain.invoke(data),
    "whatsapp": whatsapp_chain.invoke(data),
    "website": website_chain.invoke(data),
    "ad": ad_chain.invoke(data),
}

print(output)
