from pydantic import BaseModel, Field
import pandas as pd
import requests
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain.tools import tool
import os
import json # Ajouté pour la sauvegarde de la mémoire

# --- Configuration de l'API Gemini ---
os.environ["GOOGLE_API_KEY"] = "AIzaSyDvtQDt4Ayl854qmkhF2H5lhnMfHPS5c6s"

# --- Outil pour trouver les agences par ville (Version avec tableau) ---

@tool
def trouver_agences_par_ville(ville: str) -> str:
    """
    Cet outil est utilisé pour trouver les agences de ventes, les bureaux et les points de retrait de Chronopost au Maroc.
    Il prend en entrée le nom d'une ville (par exemple "Casablanca" ou "Rabat") et retourne un tableau des agences,
    avec leurs noms, adresses et numéros de téléphone.
    """
    ville_formatee = ville.capitalize()
    url = f"http://chronopost.ma:9595/chronomobile/getBranchByCity/{ville_formatee}"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, headers=headers, json={}, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        agences = data.get("cities", [])

        if not agences:
            return f"❌ Aucune agence trouvée pour la ville de {ville}. Veuillez vérifier l'orthographe ou essayer une autre ville."

        # Création d'une liste de dictionnaires pour le DataFrame
        liste_agences = []
        for agence in agences:
            liste_agences.append({
                "Nom": agence.get("name", "N/A"),
                "Adresse": agence.get("address", "N/A").replace('\n', ', '),
                "Téléphone": agence.get("tel", "N/A")
            })

        # Création du DataFrame pandas
        df = pd.DataFrame(liste_agences)
        
        # Nettoyage et formatage du DataFrame
        # Remplacer les valeurs null par des chaînes vides pour un affichage plus propre
        df = df.fillna('N/A')
        
        # Convertir le DataFrame en une chaîne de caractères au format Markdown pour un affichage structuré
        resultat = f"✅ Voici les agences Chronopost trouvées à **{ville_formatee}** :\n\n"
        resultat += df.to_markdown(index=False)
        
        return resultat
    
    except requests.exceptions.HTTPError as err:
        return f"🚨 Erreur HTTP lors de la connexion à l'API : {err}"
    except requests.exceptions.RequestException as err:
        return f"🚨 Erreur de connexion : {err}"
    except (ValueError, KeyError) as err:
        return f"🚨 Erreur lors du traitement des données de l'API : {err}"

# --- Outil de suivi de colis (corrigé avec en-tête) ---
@tool
def suivre_colis(numero_de_suivi: str) -> str:
    """
    Recherche le statut d'un colis Chronopost. Prend en entrée un numéro de suivi. 
    Retourne le dernier statut du colis en une phrase simple et lisible.
    """
    url = f"http://ws.chronopost.ma:8080/wscima/tracking/parcels?parcelIds={numero_de_suivi}"
    
    # Ajout d'un en-tête User-Agent pour simuler une requête de navigateur
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data and isinstance(data, list) and len(data) > 0:
            parcel = data[0]
            if 'events' in parcel and parcel['events']:
                last_event = parcel['events'][0]
                status_description = last_event.get('description')
                
                if status_description:
                    return f"Le dernier statut de votre colis {numero_de_suivi} est : **{status_description}**."
                else:
                    return "Aucune description de statut n'a été trouvée pour ce colis."
            else:
                return "Aucun événement de suivi n'a été trouvé pour ce numéro de colis."
        else:
            return "Numéro de suivi non trouvé dans la base de données de Chronopost."

    except requests.exceptions.HTTPError as err:
        return f"Erreur HTTP : {err}"
    except requests.exceptions.RequestException as err:
        return f"Erreur de connexion : {err}"
    except (ValueError, KeyError) as err:
        return f"Erreur de traitement des données de l'API : {err}"

# --- Outil RAG pour les questions sur Chronopost ---
@tool
def repondre_avec_documents(query: str) -> str:
    """
    Répond à des questions spécifiques sur les services, les valeurs ou les procédures de Chronopost Maroc en utilisant la documentation interne.
    Cet outil est fait pour les questions dont la réponse est probablement dans la FAQ.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    retriever = db.as_retriever()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever, memory=memory)
    response = qa_chain.invoke({"question": query})
    return response['answer']

# --- Outil de discussion générale ---
@tool
def repondre_general(query: str) -> str:
    """
    Répond à des questions qui ne sont pas liées à Chronopost, comme des salutations ou des questions générales.
    Cet outil utilise les connaissances générales du modèle pour répondre.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    return llm.invoke(query).content



# --- Chargement et préparation des données ---
try:
    df = pd.read_csv('faq.csv')
    print("Fichier CSV chargé avec succès.")
    df.dropna(inplace=True)
except FileNotFoundError:
    print("Erreur : Le fichier 'faq.csv' n'a pas été trouvé.")
    exit()
except KeyError:
    print("Erreur : Le fichier CSV ne contient pas les colonnes 'Question' ou 'Réponse'.")
    print("Veuillez vérifier les en-têtes de votre fichier.")
    exit()

documents = []
for index, row in df.iterrows():
    content = f"Question : {row['Question']}\nRéponse : {row['Réponse']}"
    metadata = {'source': 'faq.csv', 'row_id': index}
    doc = Document(page_content=content, metadata=metadata)
    documents.append(doc)

# --- Ajout du texte détaillé ---
nouveau_texte_chronopost = """
Bien sûr, voici des informations détaillées sur Chronopost Maroc :
📦 Présentation de Chronopost Maroc
Chronopost International Maroc est une filiale commune du groupe Barid Al Maghrib (BAM) et de Geopost S.A. (France), créée en mars 2001. Elle est spécialisée dans le transport et la messagerie internationale express, offrant des services de livraison rapide vers plus de 230 pays et territoires. Son siège est situé à Casablanca, au 110 boulevard Mohamed Zerktouni.
🌍 Réseau et couverture
Réseau national : Plus de 600 points de vente au Maroc, incluant 10 agences en propre et plus de 500 points de vente partenaires tels que Barid Al Maghrib, Amana, Al Barid Bank et Barid Cash.
Réseau international : Partenaire du groupe DPDgroup, Chronopost Maroc bénéficie d'un réseau mondial de plus de 32 filiales, desservant plus de 230 pays et territoires.
🚚 Services proposés
1. Chrono EXPRESS
Description : Service de messagerie internationale express garantissant la livraison en 1 à 3 jours ouvrés vers les principaux pays.
Avantages :
Livraison à domicile avec 3 tentatives.
Suivi digitalisé via le site web, l'application mobile et notifications par SMS.
Assurance optionnelle.
Services supplémentaires tels que DDP (Delivery Duty Paid), preuve de livraison, et emballage.
2. EMS (Express Mail Service)
Description : Service de messagerie internationale postale rapide assurant la livraison de vos colis à l’étranger dans des délais fiables à des tarifs économiques.
Avantages :
Délais de livraison de 5 à 10 jours ouvrés.
Territoires desservis : Plus de 100 pays et territoires.
Poids autorisé : Jusqu'à 30 kg.
Suivi digitalisé de la traçabilité des envois.
Assurance optionnelle.
Service client dédié 6j/7.
💰 Tarification
Les tarifs varient en fonction du poids, de la destination et du service choisi. Pour obtenir une estimation précise, vous pouvez utiliser l'outil de calcul de tarif disponible sur le site officiel de Chronopost Maroc.
📍 Réseau de vente
Chronopost dispose d'un réseau de vente couvrant toutes les régions du Maroc, avec des horaires d'ouverture élargis de 8h00 à 20h00, 6 jours sur 7. Vous pouvez localiser l'agence la plus proche de chez vous en consultant la carte interactive sur leur site.
🛠️ Préparation des envois
Pour assurer une expédition sans encombre, il est recommandé de :
Bien préparer votre envoi : Détailler le contenu de l’envoi, inscrire la valeur du contenu en spécifiant l’unité monétaire, et ne jamais indiquer zéro (0) comme valeur du contenu s’il s’agit d’une marchandise.
Emballer correctement : Utiliser un emballage adapté pour protéger le contenu.
Compléter les formalités douanières : Remplir correctement la lettre de transport et s'assurer de la conformité des documents.
📞 Contact
Service clients : (+212) 522 20 21 21
Email : s-client@chronopost.ma
Adresse : 110, bd Mohamed Zerktouni, 20000 Casablanca
Pour plus d'informations ou pour expédier un colis, vous pouvez visiter le site officiel de Chronopost Maroc : https://www.chronopost.ma.
"""
documents.append(Document(page_content=nouveau_texte_chronopost, metadata={'source': 'chronopost_details.txt'}))

print(f"\n{len(documents)} documents (y compris le nouveau texte) créés et prêts.")

# --- Création de la base de données vectorielle (RAG) ---
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
db = Chroma.from_documents(texts, embeddings)
print(f"Base de données vectorielle Chroma créée avec {len(texts)} morceaux de texte.")



# --- Fonction d'apprentissage RAG (ajoutée ici) ---
def apprendre_de_la_conversation(historique_conversation: list, db: Chroma):
    """
    Extrait les leçons d'une conversation et les ajoute à la base de données RAG.
    Cela permet à l'agent d'améliorer son comportement pour les futures sessions.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    
    prompt_resume = f"""
    En tant qu'assistant de Chronopost, résume les leçons apprises de la conversation suivante avec un utilisateur. 
    Concentrez-vous sur la manière dont les questions ont été posées, le ton de l'utilisateur, et comment la réponse pourrait être améliorée. 
    Par exemple, si l'utilisateur a été frustré, une leçon pourrait être "Adopter un ton plus empathique". Si l'utilisateur a posé une question complexe, une leçon pourrait être "Donner des explications par étapes".

    Conversation :
    {historique_conversation}

    Leçons apprises (sous forme de points concis) :
    """
    
    try:
        lecons = llm.invoke(prompt_resume).content
        doc_apprentissage = Document(
            page_content=f"Leçons apprises d'une conversation :\n{lecons}",
            metadata={'source': 'apprentissage_continu'}
        )
        db.add_documents([doc_apprentissage])
        print("✅ Leçons apprises ajoutées à la base de données RAG.")
    except Exception as e:
        print(f"⚠️ Erreur lors de l'apprentissage de la conversation : {e}")


# --- Création de l'agent et de sa logique ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

tools = [repondre_avec_documents, repondre_general, suivre_colis, trouver_agences_par_ville ]

# On ajoute la mémoire de conversation à l'agent lui-même
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# On initialise l'Agent Executor avec un type d'agent conversationnel
agent = initialize_agent(
    [
        repondre_avec_documents, 
        repondre_general, 
        suivre_colis, 
        trouver_agences_par_ville
    ],
    llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory, 
    handle_parsing_errors=True
)


# --- Ajout de l'interface web (Flask) ---
from flask import Flask, request, jsonify
from langchain.memory import ConversationBufferMemory
import threading

app = Flask(__name__)

# Une base de données simple pour stocker les mémoires de session
# En production, vous utiliseriez une vraie base de données ou un cache (ex: Redis)
session_memories = {}

# Cette route gère l'envoi de messages
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get('question')
    session_id = data.get('session_id')

    if not session_id:
        return jsonify({"error": "Session ID is required"}), 400

    # Récupérer ou créer la mémoire de session pour l'utilisateur
    if session_id not in session_memories:
        session_memories[session_id] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    memory = session_memories[session_id]
    
    # L'agent est réinitialisé avec la mémoire de session de l'utilisateur
    agent = initialize_agent(
        tools,  # `tools` est défini plus haut dans votre script
        llm,    # `llm` est défini plus haut dans votre script
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory, 
        handle_parsing_errors=True
    )
    
    try:
        response = agent.run(question)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Cette route gère la fin de session et déclenche l'apprentissage
@app.route('/end_session', methods=['POST'])
def end_session():
    data = request.json
    session_id = data.get('session_id')

    if session_id in session_memories:
        historique_conversation = session_memories[session_id].chat_memory.messages
        
        # Lancer l'apprentissage dans un thread séparé pour ne pas bloquer le serveur
        # `db` est la base de données ChromaDB définie dans votre code
        thread_apprentissage = threading.Thread(target=apprendre_de_la_conversation, args=(historique_conversation, db))
        thread_apprentissage.start()
        
        # Supprimer la mémoire de la session pour la libérer
        del session_memories[session_id]

    return jsonify({"status": "session ended and learning started"})

# Point d'entrée pour lancer le serveur Flask
if __name__ == '__main__':
    print("--- Serveur de l'agent Chronopost démarré ! ---")
    app.run(host='0.0.0.0', port=5000)# --- Ajout de l'interface web (Flask) ---
from flask import Flask, request, jsonify
from langchain.memory import ConversationBufferMemory
import threading

app = Flask(__name__)

# Une base de données simple pour stocker les mémoires de session
# En production, vous utiliseriez une vraie base de données ou un cache (ex: Redis)
session_memories = {}

# Cette route gère l'envoi de messages
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get('question')
    session_id = data.get('session_id')

    if not session_id:
        return jsonify({"error": "Session ID is required"}), 400

    # Récupérer ou créer la mémoire de session pour l'utilisateur
    if session_id not in session_memories:
        session_memories[session_id] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    memory = session_memories[session_id]
    
    # L'agent est réinitialisé avec la mémoire de session de l'utilisateur
    agent = initialize_agent(
        tools,  # `tools` est défini plus haut dans votre script
        llm,    # `llm` est défini plus haut dans votre script
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory, 
        handle_parsing_errors=True
    )
    
    try:
        response = agent.run(question)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Cette route gère la fin de session et déclenche l'apprentissage
@app.route('/end_session', methods=['POST'])
def end_session():
    data = request.json
    session_id = data.get('session_id')

    if session_id in session_memories:
        historique_conversation = session_memories[session_id].chat_memory.messages
        
        # Lancer l'apprentissage dans un thread séparé pour ne pas bloquer le serveur
        # `db` est la base de données ChromaDB définie dans votre code
        thread_apprentissage = threading.Thread(target=apprendre_de_la_conversation, args=(historique_conversation, db))
        thread_apprentissage.start()
        
        # Supprimer la mémoire de la session pour la libérer
        del session_memories[session_id]

    return jsonify({"status": "session ended and learning started"})

# Point d'entrée pour lancer le serveur Flask
if __name__ == '__main__':
    print("--- Serveur de l'agent Chronopost démarré ! ---")
    app.run(host='0.0.0.0', port=5000)
