"""
Ontology Reasoning Module

Implements OWL 2 RL materialization using owlrl (open source).
This enables deductive inference over the knowledge graph.
"""

from typing import List, Tuple, Set, Optional

class OntologyReasoner:
    """
    OWL 2 RL reasoner for knowledge graph materialization.
    
    Uses owlrl library (Apache 2.0 license) as free alternative to RDFox.
    """
    
    def __init__(self):
        """Initialize the ontology reasoner."""
        self.graph = None
        self.inferred_triples: List[Tuple[str, str, str]] = []
    
    def load_sample(self):
        """Load a sample ontology for demonstration."""
        try:
            from rdflib import Graph, Namespace, RDF, RDFS, OWL
            
            self.graph = Graph()
            
            # Define namespaces
            EX = Namespace("http://example.org/")
            
            # Add sample TBox (schema)
            # Scientist subClassOf Person
            self.graph.add((EX.Scientist, RDFS.subClassOf, EX.Person))
            # Physicist subClassOf Scientist
            self.graph.add((EX.Physicist, RDFS.subClassOf, EX.Scientist))
            # Nobel_Laureate subClassOf Person
            self.graph.add((EX.Nobel_Laureate, RDFS.subClassOf, EX.Person))
            
            # worksAt domain/range
            self.graph.add((EX.worksAt, RDFS.domain, EX.Person))
            self.graph.add((EX.worksAt, RDFS.range, EX.Organization))
            
            # Add sample ABox (instances)
            self.graph.add((EX.Einstein, RDF.type, EX.Physicist))
            self.graph.add((EX.Einstein, EX.worksAt, EX.Princeton))
            self.graph.add((EX.Einstein, RDF.type, EX.Nobel_Laureate))
            
            self.graph.add((EX.Curie, RDF.type, EX.Physicist))
            self.graph.add((EX.Curie, RDF.type, EX.Nobel_Laureate))
            
            print(f"Loaded ontology with {len(self.graph)} triples")
            
        except ImportError:
            print("rdflib not installed. Using mock ontology.")
            self.graph = None
    
    def materialize(self, max_iterations: int = 10) -> List[Tuple[str, str, str]]:
        """
        Perform OWL 2 RL materialization.
        
        This computes the deductive closure of the ontology.
        
        Args:
            max_iterations: Maximum reasoning iterations
            
        Returns:
            List of newly inferred triples
        """
        if self.graph is None:
            # Mock materialization for demo
            self.inferred_triples = [
                ("Einstein", "type", "Scientist"),
                ("Einstein", "type", "Person"),
                ("Curie", "type", "Scientist"),
                ("Curie", "type", "Person"),
                ("Princeton", "type", "Organization")
            ]
            return self.inferred_triples
        
        try:
            import owlrl
            
            original_size = len(self.graph)
            
            # Apply OWL 2 RL reasoning
            owlrl.DeductiveClosure(
                owlrl.OWLRL_Semantics,
                rdfs_closure=True,
                axiomatic_triples=False
            ).expand(self.graph)
            
            new_size = len(self.graph)
            
            print(f"Materialization complete: {original_size} → {new_size} triples")
            print(f"Inferred {new_size - original_size} new triples")
            
            # Extract inferred triples (simplified)
            self.inferred_triples = []
            for s, p, o in self.graph:
                self.inferred_triples.append((str(s), str(p), str(o)))
            
            return self.inferred_triples[-10:]  # Return sample
            
        except ImportError:
            print("owlrl not installed. Using mock reasoning.")
            return self.materialize()
    
    def query(self, sparql: str) -> List[dict]:
        """Execute a SPARQL query on the materialized graph."""
        if self.graph is None:
            return []
        
        try:
            results = self.graph.query(sparql)
            return [dict(row) for row in results]
        except Exception as e:
            print(f"Query error: {e}")
            return []


"""
THEOREM 4: OWL 2 RL Materialization Completeness

OWL 2 RL is a tractable profile of OWL 2 that supports:
- Subclass hierarchies (rdfs:subClassOf)
- Property hierarchies (rdfs:subPropertyOf)
- Domain/range reasoning
- Transitivity, symmetry, reflexivity
- Some/all values from (with restrictions)

PROPERTIES:
1. SOUNDNESS: All inferred triples are logical consequences
2. COMPLETENESS: For OWL 2 RL expressivity, all consequences are inferred
3. TRACTABILITY: Materialization is polynomial in input size

COMPLEXITY:
- Time: O(n³) worst case, O(n²) typical for sparse ontologies
- Space: O(n²) for storing inferred triples

This is why we use OWL 2 RL instead of full OWL 2:
- Full OWL 2 is undecidable
- OWL 2 RL gives useful inference in polynomial time

REFERENCE: Motik et al., "OWL 2 Web Ontology Language Profiles"
"""


# Example rules implemented by OWL 2 RL
OWL2RL_RULES = """
# Subclass transitivity (rdfs11)
IF (?x rdfs:subClassOf ?y) AND (?y rdfs:subClassOf ?z)
THEN (?x rdfs:subClassOf ?z)

# Instance inheritance (rdfs9) 
IF (?x rdf:type ?c) AND (?c rdfs:subClassOf ?d)
THEN (?x rdf:type ?d)

# Domain inference (rdfs2)
IF (?x ?p ?y) AND (?p rdfs:domain ?c)
THEN (?x rdf:type ?c)

# Range inference (rdfs3)
IF (?x ?p ?y) AND (?p rdfs:range ?c)
THEN (?y rdf:type ?c)
"""
