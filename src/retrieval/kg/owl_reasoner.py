"""OWL Ontology reasoning for semantic expansion.

This module implements OWL-based ontological reasoning to expand
entity sets using:
1. Subsumption (class hierarchy)
2. Property inheritance
3. Transitive relations
4. Inference rules

Uses OWLReady2 for OWL reasoning with Pellet/HermiT reasoners.
"""

from typing import List, Dict, Set, Optional, Any
import logging

logger = logging.getLogger(__name__)

try:
    from owlready2 import (
        get_ontology, 
        sync_reasoner_pellet,
        sync_reasoner_hermit,
        Thing,
        ObjectProperty,
        TransitiveProperty,
        default_world
    )
    HAS_OWLREADY = True
except ImportError:
    HAS_OWLREADY = False
    logger.warning("owlready2 not installed. OWL reasoning disabled.")


class OWLReasoner:
    """OWL reasoner for ontological inference.
    
    This class wraps OWLReady2 to provide ontological reasoning
    capabilities for entity expansion.
    
    Attributes:
        ontology: The loaded OWL ontology.
        entity_to_classes: Mapping from entities to their OWL classes.
        class_hierarchy: Mapping from classes to their superclasses.
    """
    
    def __init__(self, ontology_path: Optional[str] = None):
        """Initialize OWL reasoner.
        
        Args:
            ontology_path: Path to OWL ontology file (.owl, .rdf).
                          If None, creates an empty ontology.
        """
        if not HAS_OWLREADY:
            raise ImportError("owlready2 is required. Install with: pip install owlready2")
        
        if ontology_path:
            self.ontology = get_ontology(ontology_path).load()
        else:
            self.ontology = get_ontology("http://example.org/pkm.owl")
        
        self.entity_to_classes: Dict[str, Set[str]] = {}
        self.class_instances: Dict[str, Set[str]] = {}
        self._reasoner_synced = False
    
    def create_pkm_ontology(self) -> None:
        """Create a basic PKM (Personal Knowledge Management) ontology.
        
        This creates a simple ontology with common PKM classes and properties.
        """
        with self.ontology:
            # Base classes
            class Document(Thing): pass
            class Person(Thing): pass
            class Organization(Thing): pass
            class Concept(Thing): pass
            class Location(Thing): pass
            class Event(Thing): pass
            class Project(Thing): pass
            class Topic(Thing): pass
            
            # Subclasses
            class Article(Document): pass
            class Paper(Document): pass
            class Note(Document): pass
            class Email(Document): pass
            
            class Researcher(Person): pass
            class Author(Person): pass
            
            class University(Organization): pass
            class Company(Organization): pass
            
            # Object properties
            class mentions(ObjectProperty):
                domain = [Document]
                range = [Thing]
            
            class authorOf(ObjectProperty):
                domain = [Person]
                range = [Document]
            
            class affiliatedWith(ObjectProperty):
                domain = [Person]
                range = [Organization]
            
            class relatedTo(ObjectProperty, TransitiveProperty):
                domain = [Concept]
                range = [Concept]
            
            class partOf(ObjectProperty, TransitiveProperty):
                domain = [Thing]
                range = [Thing]
            
            class locatedIn(ObjectProperty, TransitiveProperty):
                domain = [Thing]
                range = [Location]
            
            class hasParticipant(ObjectProperty):
                domain = [Event]
                range = [Person]
        
        logger.info("Created PKM ontology with base classes and properties")
    
    def add_entity(
        self, 
        entity_id: str, 
        entity_label: str,
        entity_types: List[str]
    ) -> None:
        """Add an entity to the ontology.
        
        Args:
            entity_id: Unique entity identifier.
            entity_label: Human-readable label.
            entity_types: List of type names (must match ontology classes).
        """
        with self.ontology:
            # Find or create the most specific class
            classes = []
            for type_name in entity_types:
                cls = self.ontology.search_one(iri=f"*{type_name}")
                if cls:
                    classes.append(cls)
            
            if not classes:
                classes = [Thing]
            
            # Create individual
            entity = classes[0](entity_id)
            entity.label = [entity_label]
            
            # Track mapping
            self.entity_to_classes[entity_id] = {c.name for c in classes}
    
    def add_relation(
        self, 
        subject_id: str, 
        predicate: str, 
        object_id: str
    ) -> None:
        """Add a relation between entities.
        
        Args:
            subject_id: Subject entity ID.
            predicate: Relation name (must match ontology property).
            object_id: Object entity ID.
        """
        with self.ontology:
            subj = self.ontology.search_one(iri=f"*{subject_id}")
            obj = self.ontology.search_one(iri=f"*{object_id}")
            prop = self.ontology.search_one(iri=f"*{predicate}")
            
            if subj and obj and prop:
                getattr(subj, prop.python_name).append(obj)
    
    def sync_reasoner(self, reasoner: str = "pellet") -> None:
        """Run the OWL reasoner to infer new facts.
        
        Args:
            reasoner: Reasoner to use ("pellet" or "hermit").
        """
        try:
            with self.ontology:
                if reasoner == "pellet":
                    sync_reasoner_pellet(infer_property_values=True)
                else:
                    sync_reasoner_hermit(infer_property_values=True)
            self._reasoner_synced = True
            logger.info(f"Reasoner {reasoner} synced successfully")
        except Exception as e:
            logger.error(f"Reasoner failed: {e}")
            self._reasoner_synced = False
    
    def get_superclasses(self, class_name: str) -> Set[str]:
        """Get all superclasses of a class.
        
        Args:
            class_name: Class name.
            
        Returns:
            Set of superclass names.
        """
        cls = self.ontology.search_one(iri=f"*{class_name}")
        if not cls:
            return set()
        
        superclasses = set()
        for ancestor in cls.ancestors():
            if ancestor != Thing and hasattr(ancestor, 'name'):
                superclasses.add(ancestor.name)
        
        return superclasses
    
    def get_subclasses(self, class_name: str) -> Set[str]:
        """Get all subclasses of a class.
        
        Args:
            class_name: Class name.
            
        Returns:
            Set of subclass names.
        """
        cls = self.ontology.search_one(iri=f"*{class_name}")
        if not cls:
            return set()
        
        subclasses = set()
        for descendant in cls.descendants():
            if descendant != cls and hasattr(descendant, 'name'):
                subclasses.add(descendant.name)
        
        return subclasses
    
    def get_instances(self, class_name: str) -> Set[str]:
        """Get all instances of a class.
        
        Args:
            class_name: Class name.
            
        Returns:
            Set of instance IDs.
        """
        cls = self.ontology.search_one(iri=f"*{class_name}")
        if not cls:
            return set()
        
        instances = set()
        for instance in cls.instances():
            if hasattr(instance, 'name'):
                instances.add(instance.name)
        
        return instances
    
    def get_related(
        self, 
        entity_id: str, 
        property_name: Optional[str] = None
    ) -> Set[str]:
        """Get entities related to the given entity.
        
        Args:
            entity_id: Entity ID.
            property_name: Optional property to filter by.
            
        Returns:
            Set of related entity IDs.
        """
        entity = self.ontology.search_one(iri=f"*{entity_id}")
        if not entity:
            return set()
        
        related = set()
        
        if property_name:
            prop = self.ontology.search_one(iri=f"*{property_name}")
            if prop:
                values = getattr(entity, prop.python_name, [])
                for val in values:
                    if hasattr(val, 'name'):
                        related.add(val.name)
        else:
            # Get all related entities via any property
            for prop in self.ontology.object_properties():
                values = getattr(entity, prop.python_name, [])
                for val in values:
                    if hasattr(val, 'name'):
                        related.add(val.name)
        
        return related
    
    def materialize(self, entity_ids: List[str]) -> List[str]:
        """Expand entity set using ontological reasoning.
        
        This performs:
        1. Add instances of the same classes
        2. Add entities related via transitive properties
        3. Add entities from inferred facts (if reasoner was run)
        
        Args:
            entity_ids: List of seed entity IDs.
            
        Returns:
            Expanded list of entity IDs.
        """
        expanded = set(entity_ids)
        
        for eid in entity_ids:
            entity = self.ontology.search_one(iri=f"*{eid}")
            if not entity:
                continue
            
            # Get entity's classes
            for cls in entity.is_a:
                if hasattr(cls, 'name'):
                    # Add sibling instances (same class)
                    for instance in cls.instances():
                        if hasattr(instance, 'name'):
                            expanded.add(instance.name)
            
            # Get related entities (especially via transitive properties)
            related = self.get_related(eid)
            expanded.update(related)
        
        return list(expanded)
    
    def explain_inference(
        self, 
        entity_id: str, 
        inferred_entity_id: str
    ) -> List[str]:
        """Explain why an entity was inferred.
        
        Args:
            entity_id: Original entity.
            inferred_entity_id: Inferred entity.
            
        Returns:
            List of explanation strings.
        """
        explanations = []
        
        entity = self.ontology.search_one(iri=f"*{entity_id}")
        inferred = self.ontology.search_one(iri=f"*{inferred_entity_id}")
        
        if not entity or not inferred:
            return ["Entities not found in ontology"]
        
        # Check class relationship
        entity_classes = set(entity.is_a)
        inferred_classes = set(inferred.is_a)
        common_classes = entity_classes & inferred_classes
        
        if common_classes:
            class_names = [c.name for c in common_classes if hasattr(c, 'name')]
            explanations.append(f"Same class: {', '.join(class_names)}")
        
        # Check property relationship
        for prop in self.ontology.object_properties():
            values = getattr(entity, prop.python_name, [])
            if inferred in values:
                explanations.append(f"Related via: {prop.name}")
        
        return explanations if explanations else ["No direct relationship found"]
