import os
import logging
from retry import retry
from neo4j import GraphDatabase

# Environment variables will be passed into the Docker container
HOSPITALS_CSV_PATH = os.getenv("HOSPITALS_CSV_PATH")
PAYERS_CSV_PATH = os.getenv("PAYERS_CSV_PATH")
PHYSICIANS_CSV_PATH = os.getenv("PHYSICIANS_CSV_PATH")
PATIENTS_CSV_PATH = os.getenv("PATIENTS_CSV_PATH")
VISITS_CSV_PATH = os.getenv("VISITS_CSV_PATH")
REVIEWS_CSV_PATH = os.getenv("REVIEWS_CSV_PATH")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

LOGGER = logging.getLogger(__name__)

# Define nodes in Graph Database
NODES = ["Hospital", "Payer", "Physician", "Patient", "Visit", "Review"]


# Create and run queries enforcing each node to have a unique ID
def _set_uniqueness_constraints(tx, node):
    query = f"""CREATE CONSTRAINT IF NOT EXISTS FOR (n:{node})
        REQUIRE n.id IS UNIQUE;"""
    _ = tx.run(query, {})


# Instantiate a driver that connects to the Neo4j instance and set uniqueness constraints for each system node
# Decorator to cope eventual failed connections
@retry(tries=100, delay=10)
def load_hospital_graph_from_csv() -> None:
    """Load structured hospital CSV data following
    a specific ontology into Neo4j"""

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    LOGGER.info("Setting uniqueness constraints on nodes")
    with driver.session(database="neo4j") as session:
        for node in NODES:
            session.execute_write(_set_uniqueness_constraints, node)

    LOGGER.info("Loading hospital nodes")
    with driver.session(database="neo4j") as session:
        query = f"""
        LOAD CSV WITH HEADERS
        FROM '{HOSPITALS_CSV_PATH}' AS hospitals
        MERGE (h:Hospital {{id: toInteger(hospitals.hospital_id),
                            name: hospitals.hospital_name,
                            state_name: hospitals.hospital_state}});
        """
        _ = session.run(query, {})

    LOGGER.info("Loading 'WRITES' relationships")
    with driver.session(database="neo4j") as session:
        query = f"""
        LOAD CSV WITH HEADERS FROM '{REVIEWS_CSV_PATH}' AS reviews
            MATCH (v:Visit {{id: toInteger(reviews.visit_id)}})
            MATCH (r:Review {{id: toInteger(reviews.review_id)}})
            MERGE (v)-[writes:WRITES]->(r)
        """
        _ = session.run(query, {})

    LOGGER.info("Loading 'TREATS' relationships")
    with driver.session(database="neo4j") as session:
        query = f"""
        LOAD CSV WITH HEADERS FROM '{VISITS_CSV_PATH}' AS visits
            MATCH (p:Physician {{id: toInteger(visits.physician_id)}})
            MATCH (v:Visit {{id: toInteger(visits.visit_id)}})
            MERGE (p)-[treats:TREATS]->(v)
        """
        _ = session.run(query, {})

    LOGGER.info("Loading 'COVERED_BY' relationships")
    with driver.session(database="neo4j") as session:
        query = f"""
        LOAD CSV WITH HEADERS FROM '{VISITS_CSV_PATH}' AS visits
            MATCH (v:Visit {{id: toInteger(visits.visit_id)}})
            MATCH (p:Payer {{id: toInteger(visits.payer_id)}})
            MERGE (v)-[covered_by:COVERED_BY]->(p)
            ON CREATE SET
                covered_by.service_date = visits.discharge_date,
                covered_by.billing_amount = toFloat(visits.billing_amount)
        """
        _ = session.run(query, {})

    LOGGER.info("Loading 'HAS' relationships")
    with driver.session(database="neo4j") as session:
        query = f"""
        LOAD CSV WITH HEADERS FROM '{VISITS_CSV_PATH}' AS visits
            MATCH (p:Patient {{id: toInteger(visits.patient_id)}})
            MATCH (v:Visit {{id: toInteger(visits.visit_id)}})
            MERGE (p)-[has:HAS]->(v)
        """
        _ = session.run(query, {})

    LOGGER.info("Loading 'EMPLOYS' relationships")
    with driver.session(database="neo4j") as session:
        query = f"""
        LOAD CSV WITH HEADERS FROM '{VISITS_CSV_PATH}' AS visits
            MATCH (h:Hospital {{id: toInteger(visits.hospital_id)}})
            MATCH (p:Physician {{id: toInteger(visits.physician_id)}})
            MERGE (h)-[employs:EMPLOYS]->(p)
        """
        _ = session.run(query, {})


if __name__ == "__main__":
    load_hospital_graph_from_csv()
