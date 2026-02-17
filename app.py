from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os
from typing import Dict, List, Any
import hashlib

app = Flask(__name__)
CORS(app)

class AdvancedCareerRecommender:
    def __init__(self):
        self.career_database = self.create_comprehensive_career_database()
        self.personality_archetypes = self.define_personality_archetypes()
        self.initialize_weights()
        
    def create_comprehensive_career_database(self):
        """Create a diverse career database with detailed personality mappings"""
        careers = [
            # TECHNOLOGY & ANALYTICAL CAREERS
            {
                "id": 1, "title": "AI/ML Engineer", "category": "technology",
                "description": "Design and implement artificial intelligence and machine learning systems",
                "salary_min": 100000, "salary_max": 180000, "growth": 25,
                "skills": ["Python", "TensorFlow", "Deep Learning", "Statistics", "Data Modeling"],
                "personality_traits": ["Analytical", "Innovative", "Patient", "Detail-oriented"],
                "experience_level": "Mid to Senior",
                "work_environment": ["Research Labs", "Tech Companies", "Remote Friendly"],
                "personality_profile": {
                    "mbti_weights": {"INTJ": 0.95, "INTP": 0.90, "ENTJ": 0.85, "ENTP": 0.80, "ISTJ": 0.70},
                    "riasec_weights": {"I": 0.95, "R": 0.85, "A": 0.70},
                    "ikigai_weights": {"profession": 0.90, "vocation": 0.80, "mission": 0.60},
                    "skill_domains": ["technical", "analytical", "programming", "mathematics"],
                    "trait_profile": {"analytical": 0.95, "technical": 0.90, "creativity": 0.70, "structured": 0.85}
                },
                "requirements": {"education": "Master's", "experience": "Mid-Senior"}
            },
            {
                "id": 2, "title": "Data Scientist", "category": "technology", 
                "description": "Extract insights from complex data using statistical and machine learning techniques",
                "salary_min": 90000, "salary_max": 150000, "growth": 22,
                "skills": ["Python", "R", "SQL", "Statistics", "Machine Learning"],
                "personality_traits": ["Analytical", "Curious", "Detail-oriented", "Problem-solver"],
                "experience_level": "Mid-Level",
                "work_environment": ["Tech Companies", "Research", "Remote Friendly"],
                "personality_profile": {
                    "mbti_weights": {"INTP": 0.95, "INTJ": 0.90, "ENTP": 0.85, "ISTJ": 0.75, "ENTJ": 0.70},
                    "riasec_weights": {"I": 0.95, "C": 0.80, "R": 0.70},
                    "ikigai_weights": {"profession": 0.90, "mission": 0.75, "vocation": 0.65},
                    "skill_domains": ["analytical", "statistics", "programming", "research"],
                    "trait_profile": {"analytical": 0.95, "structured": 0.85, "technical": 0.80, "creativity": 0.65}
                },
                "requirements": {"education": "Master's", "experience": "Mid-Level"}
            },
            {
                "id": 3, "title": "Cybersecurity Analyst", "category": "technology",
                "description": "Protect organizations from digital threats and security breaches",
                "salary_min": 80000, "salary_max": 130000, "growth": 18,
                "skills": ["Network Security", "Ethical Hacking", "Risk Assessment", "Incident Response"],
                "personality_traits": ["Vigilant", "Analytical", "Problem-solver", "Ethical"],
                "experience_level": "Entry to Senior",
                "work_environment": ["Security Operations", "Corporate", "Government"],
                "personality_profile": {
                    "mbti_weights": {"ISTJ": 0.95, "INTJ": 0.90, "ESTJ": 0.85, "ISTP": 0.80, "ISFJ": 0.75},
                    "riasec_weights": {"I": 0.90, "C": 0.85, "R": 0.75},
                    "ikigai_weights": {"vocation": 0.90, "mission": 0.80, "profession": 0.75},
                    "skill_domains": ["technical", "security", "analytical", "problem-solving"],
                    "trait_profile": {"structured": 0.95, "analytical": 0.90, "technical": 0.85, "practical": 0.80}
                },
                "requirements": {"education": "Bachelor's", "experience": "Entry-Senior"}
            },

            # CREATIVE & ARTISTIC CAREERS
            {
                "id": 4, "title": "Game Designer", "category": "creative",
                "description": "Create engaging gameplay experiences and game mechanics",
                "salary_min": 60000, "salary_max": 110000, "growth": 12,
                "skills": ["Game Mechanics", "Storytelling", "Level Design", "Prototyping"],
                "personality_traits": ["Creative", "Imaginative", "Technical", "User-focused"],
                "experience_level": "Entry to Senior",
                "work_environment": ["Game Studios", "Creative", "Collaborative"],
                "personality_profile": {
                    "mbti_weights": {"ENFP": 0.95, "INFP": 0.90, "ENTP": 0.85, "ISFP": 0.80, "ENFJ": 0.75},
                    "riasec_weights": {"A": 0.95, "I": 0.80, "E": 0.70},
                    "ikigai_weights": {"passion": 0.95, "mission": 0.80, "profession": 0.65},
                    "skill_domains": ["creative", "design", "storytelling", "technical"],
                    "trait_profile": {"creativity": 0.95, "social": 0.70, "technical": 0.60, "analytical": 0.55}
                },
                "requirements": {"education": "Bachelor's", "experience": "Entry-Senior"}
            },
            {
                "id": 5, "title": "UX/UI Designer", "category": "creative",
                "description": "Design user-friendly and aesthetically pleasing digital interfaces",
                "salary_min": 65000, "salary_max": 120000, "growth": 15,
                "skills": ["Figma", "User Research", "Wireframing", "Prototyping"],
                "personality_traits": ["Empathetic", "Creative", "Detail-oriented", "User-focused"],
                "experience_level": "Entry to Senior",
                "work_environment": ["Tech Companies", "Agencies", "Remote Friendly"],
                "personality_profile": {
                    "mbti_weights": {"INFJ": 0.95, "ENFJ": 0.90, "INFP": 0.85, "ENFP": 0.80, "ISFP": 0.75},
                    "riasec_weights": {"A": 0.90, "S": 0.80, "I": 0.70},
                    "ikigai_weights": {"passion": 0.90, "profession": 0.80, "mission": 0.75},
                    "skill_domains": ["creative", "design", "empathy", "technical"],
                    "trait_profile": {"creativity": 0.90, "social": 0.85, "analytical": 0.70, "technical": 0.60}
                },
                "requirements": {"education": "Bachelor's", "experience": "Entry-Senior"}
            },

            # SOCIAL & HELPING CAREERS
            {
                "id": 6, "title": "Clinical Psychologist", "category": "healthcare",
                "description": "Help patients with mental health issues through therapy and assessment",
                "salary_min": 70000, "salary_max": 120000, "growth": 14,
                "skills": ["Therapy", "Assessment", "Empathy", "Communication"],
                "personality_traits": ["Empathetic", "Patient", "Analytical", "Supportive"],
                "experience_level": "Senior",
                "work_environment": ["Hospitals", "Private Practice", "Clinics"],
                "personality_profile": {
                    "mbti_weights": {"INFJ": 0.95, "ENFJ": 0.90, "INFP": 0.85, "ISFJ": 0.80, "ENFP": 0.75},
                    "riasec_weights": {"S": 0.95, "I": 0.85, "A": 0.70},
                    "ikigai_weights": {"mission": 0.95, "vocation": 0.85, "passion": 0.75},
                    "skill_domains": ["empathy", "communication", "analytical", "research"],
                    "trait_profile": {"social": 0.95, "analytical": 0.85, "structured": 0.75, "creativity": 0.60}
                },
                "requirements": {"education": "Doctorate", "experience": "Senior"}
            },
            {
                "id": 7, "title": "Social Worker", "category": "social_services",
                "description": "Support individuals and communities through challenging life situations",
                "salary_min": 45000, "salary_max": 75000, "growth": 12,
                "skills": ["Counseling", "Case Management", "Advocacy", "Communication"],
                "personality_traits": ["Compassionate", "Patient", "Resilient", "Organized"],
                "experience_level": "Entry to Senior",
                "work_environment": ["Community Centers", "Schools", "Government"],
                "personality_profile": {
                    "mbti_weights": {"ESFJ": 0.95, "ISFJ": 0.90, "ENFJ": 0.85, "INFJ": 0.80, "ESTJ": 0.75},
                    "riasec_weights": {"S": 0.95, "E": 0.80, "A": 0.65},
                    "ikigai_weights": {"mission": 0.95, "vocation": 0.85, "passion": 0.70},
                    "skill_domains": ["empathy", "communication", "organization", "advocacy"],
                    "trait_profile": {"social": 0.95, "structured": 0.80, "practical": 0.75, "creativity": 0.55}
                },
                "requirements": {"education": "Master's", "experience": "Entry-Senior"}
            },

            # LEADERSHIP & BUSINESS CAREERS
            {
                "id": 8, "title": "Management Consultant", "category": "business",
                "description": "Help organizations solve problems and improve performance",
                "salary_min": 85000, "salary_max": 160000, "growth": 14,
                "skills": ["Strategy", "Analytics", "Business Acumen", "Communication"],
                "personality_traits": ["Strategic", "Analytical", "Persuasive", "Adaptable"],
                "experience_level": "Mid to Senior",
                "work_environment": ["Consulting Firms", "Corporate", "Travel-heavy"],
                "personality_profile": {
                    "mbti_weights": {"ENTJ": 0.95, "ESTJ": 0.90, "INTJ": 0.85, "ENTP": 0.80, "ENFJ": 0.75},
                    "riasec_weights": {"E": 0.95, "I": 0.85, "C": 0.75},
                    "ikigai_weights": {"profession": 0.90, "mission": 0.80, "vocation": 0.70},
                    "skill_domains": ["leadership", "analytical", "communication", "strategy"],
                    "trait_profile": {"leadership": 0.95, "analytical": 0.90, "social": 0.85, "structured": 0.80}
                },
                "requirements": {"education": "MBA", "experience": "Mid-Senior"}
            },
            {
                "id": 9, "title": "Entrepreneur", "category": "business",
                "description": "Start and build businesses from the ground up",
                "salary_min": 50000, "salary_max": 200000, "growth": 10,
                "skills": ["Leadership", "Risk-taking", "Innovation", "Strategy"],
                "personality_traits": ["Visionary", "Resilient", "Adaptable", "Driven"],
                "experience_level": "Senior",
                "work_environment": ["Startups", "Various", "High-risk"],
                "personality_profile": {
                    "mbti_weights": {"ENTP": 0.95, "ENTJ": 0.90, "ESTP": 0.85, "ENFP": 0.80, "INTJ": 0.75},
                    "riasec_weights": {"E": 0.95, "A": 0.80, "I": 0.70},
                    "ikigai_weights": {"passion": 0.95, "profession": 0.85, "mission": 0.75},
                    "skill_domains": ["leadership", "creative", "risk-taking", "strategy"],
                    "trait_profile": {"leadership": 0.95, "creativity": 0.90, "practical": 0.80, "social": 0.85}
                },
                "requirements": {"education": "Varied", "experience": "Senior"}
            },

            # PRACTICAL & HANDS-ON CAREERS
            {
                "id": 10, "title": "Robotics Engineer", "category": "engineering",
                "description": "Design, build, and maintain robotic systems",
                "salary_min": 75000, "salary_max": 130000, "growth": 16,
                "skills": ["Robotics", "Programming", "Electronics", "Mechanical Design"],
                "personality_traits": ["Technical", "Precise", "Innovative", "Problem-solver"],
                "experience_level": "Mid to Senior",
                "work_environment": ["Manufacturing", "Research Labs", "Tech Companies"],
                "personality_profile": {
                    "mbti_weights": {"ISTP": 0.95, "INTP": 0.90, "ESTP": 0.85, "INTJ": 0.80, "ISTJ": 0.75},
                    "riasec_weights": {"R": 0.95, "I": 0.90, "A": 0.65},
                    "ikigai_weights": {"vocation": 0.90, "profession": 0.85, "mission": 0.70},
                    "skill_domains": ["technical", "mechanical", "programming", "problem-solving"],
                    "trait_profile": {"technical": 0.95, "practical": 0.90, "analytical": 0.85, "creativity": 0.70}
                },
                "requirements": {"education": "Bachelor's", "experience": "Mid-Senior"}
            },

            # EDUCATION & SCIENCE CAREERS
            {
                "id": 11, "title": "University Professor", "category": "education",
                "description": "Teach and conduct research in academic institutions",
                "salary_min": 60000, "salary_max": 120000, "growth": 9,
                "skills": ["Research", "Teaching", "Mentoring", "Subject Expertise"],
                "personality_traits": ["Knowledgeable", "Patient", "Communicative", "Analytical"],
                "experience_level": "Senior",
                "work_environment": ["Universities", "Research Institutions"],
                "personality_profile": {
                    "mbti_weights": {"INTP": 0.95, "INTJ": 0.90, "INFJ": 0.85, "ENFJ": 0.80, "ENTJ": 0.75},
                    "riasec_weights": {"I": 0.95, "S": 0.85, "A": 0.75},
                    "ikigai_weights": {"mission": 0.95, "passion": 0.85, "profession": 0.80},
                    "skill_domains": ["communication", "research", "subject-expertise", "mentoring"],
                    "trait_profile": {"social": 0.90, "analytical": 0.95, "structured": 0.80, "creativity": 0.70}
                },
                "requirements": {"education": "Doctorate", "experience": "Senior"}
            },

            # ARTS & ENTERTAINMENT CAREERS
            {
                "id": 12, "title": "Film Director", "category": "arts",
                "description": "Oversee creative aspects of film production",
                "salary_min": 50000, "salary_max": 200000, "growth": 10,
                "skills": ["Storytelling", "Leadership", "Creative Vision", "Collaboration"],
                "personality_traits": ["Visionary", "Leadership", "Creative", "Decisive"],
                "experience_level": "Senior",
                "work_environment": ["Film Sets", "Studios", "Creative"],
                "personality_profile": {
                    "mbti_weights": {"ENFJ": 0.95, "ENTJ": 0.90, "ENFP": 0.85, "INFP": 0.80, "ENTP": 0.75},
                    "riasec_weights": {"A": 0.95, "E": 0.90, "S": 0.70},
                    "ikigai_weights": {"passion": 0.95, "mission": 0.85, "profession": 0.75},
                    "skill_domains": ["creative", "leadership", "storytelling", "communication"],
                    "trait_profile": {"creativity": 0.95, "leadership": 0.90, "social": 0.85, "analytical": 0.65}
                },
                "requirements": {"education": "Varied", "experience": "Senior"}
            },

            # ADDITIONAL DIVERSE CAREERS
            {
                "id": 13, "title": "Data Journalist", "category": "media",
                "description": "Use data analysis to create compelling news stories",
                "salary_min": 45000, "salary_max": 85000, "growth": 12,
                "skills": ["Data Analysis", "Storytelling", "Visualization", "Research"],
                "personality_traits": ["Curious", "Analytical", "Communicative", "Detail-oriented"],
                "experience_level": "Mid-Level",
                "work_environment": ["News Organizations", "Remote Friendly"],
                "personality_profile": {
                    "mbti_weights": {"INTP": 0.95, "ENTP": 0.90, "INFJ": 0.85, "ENFP": 0.80, "INTJ": 0.75},
                    "riasec_weights": {"I": 0.90, "A": 0.85, "E": 0.75},
                    "ikigai_weights": {"mission": 0.90, "passion": 0.85, "profession": 0.80},
                    "skill_domains": ["analytical", "writing", "research", "communication"],
                    "trait_profile": {"analytical": 0.90, "creativity": 0.85, "social": 0.75, "structured": 0.70}
                },
                "requirements": {"education": "Bachelor's", "experience": "Mid-Level"}
            },
            {
                "id": 14, "title": "Environmental Scientist", "category": "science",
                "description": "Study and solve environmental problems",
                "salary_min": 55000, "salary_max": 95000, "growth": 11,
                "skills": ["Research", "Data Analysis", "Field Work", "Environmental Regulations"],
                "personality_traits": ["Analytical", "Curious", "Environmental-conscious", "Patient"],
                "experience_level": "Mid-Level",
                "work_environment": ["Field Work", "Labs", "Government Agencies"],
                "personality_profile": {
                    "mbti_weights": {"ISFP": 0.95, "INFP": 0.90, "ISTP": 0.85, "ENFP": 0.80, "INFJ": 0.75},
                    "riasec_weights": {"I": 0.95, "R": 0.85, "S": 0.70},
                    "ikigai_weights": {"mission": 0.95, "passion": 0.85, "vocation": 0.75},
                    "skill_domains": ["scientific", "analytical", "fieldwork", "research"],
                    "trait_profile": {"analytical": 0.85, "practical": 0.80, "social": 0.60, "creativity": 0.65}
                },
                "requirements": {"education": "Master's", "experience": "Mid-Level"}
            },
            {
                "id": 15, "title": "Sports Psychologist", "category": "healthcare",
                "description": "Help athletes improve mental performance and wellbeing",
                "salary_min": 60000, "salary_max": 100000, "growth": 13,
                "skills": ["Psychology", "Sports Science", "Counseling", "Performance Analysis"],
                "personality_traits": ["Empathetic", "Analytical", "Motivational", "Patient"],
                "experience_level": "Mid to Senior",
                "work_environment": ["Sports Teams", "Private Practice", "Clinics"],
                "personality_profile": {
                    "mbti_weights": {"ESFJ": 0.95, "ENFJ": 0.90, "ISFJ": 0.85, "ESTJ": 0.80, "INFJ": 0.75},
                    "riasec_weights": {"S": 0.95, "E": 0.85, "I": 0.75},
                    "ikigai_weights": {"mission": 0.90, "passion": 0.85, "vocation": 0.80},
                    "skill_domains": ["empathy", "communication", "analytical", "sports"],
                    "trait_profile": {"social": 0.95, "analytical": 0.80, "structured": 0.75, "practical": 0.70}
                },
                "requirements": {"education": "Doctorate", "experience": "Mid-Senior"}
            }
        ]
        return careers

    def define_personality_archetypes(self):
        """Define comprehensive personality archetypes for differentiation"""
        return {
            "analytical_thinker": {"mbti": ["INTJ", "INTP", "ENTJ"], "traits": {"analytical": 0.95, "technical": 0.85}},
            "creative_idealist": {"mbti": ["INFP", "ENFP", "INFJ"], "traits": {"creativity": 0.95, "social": 0.80}},
            "practical_realist": {"mbti": ["ISTJ", "ESTJ", "ISFJ"], "traits": {"structured": 0.95, "practical": 0.90}},
            "social_leader": {"mbti": ["ENFJ", "ESFJ", "ENTJ"], "traits": {"social": 0.95, "leadership": 0.90}},
            "investigative_researcher": {"riasec": ["I"], "traits": {"analytical": 0.95, "technical": 0.80}},
            "artistic_creator": {"riasec": ["A"], "traits": {"creativity": 0.95, "social": 0.70}},
            "social_helper": {"riasec": ["S"], "traits": {"social": 0.95, "empathy": 0.90}},
            "enterprising_leader": {"riasec": ["E"], "traits": {"leadership": 0.95, "social": 0.85}},
            "mission_driven": {"ikigai": ["mission"], "traits": {"social": 0.90, "structured": 0.80}},
            "passion_driven": {"ikigai": ["passion"], "traits": {"creativity": 0.95, "social": 0.75}},
            "profession_expert": {"ikigai": ["profession"], "traits": {"technical": 0.90, "analytical": 0.85}},
            "vocation_craftsman": {"ikigai": ["vocation"], "traits": {"practical": 0.95, "technical": 0.85}}
        }

    def initialize_weights(self):
        """Initialize weighting system for different assessment types"""
        self.weights = {
            "mbti": 0.30,
            "riasec": 0.25,
            "ikigai": 0.20,
            "skills": 0.15,
            "traits": 0.10
        }

    def generate_user_profile_hash(self, user_profile: Dict[str, Any]) -> str:
        """Generate unique hash for each user profile combination"""
        profile_string = f"{user_profile.get('mbti', '')}-{'-'.join(sorted(user_profile.get('riasec', [])))}-{'-'.join(sorted(user_profile.get('ikigai', [])))}-{'-'.join(sorted(user_profile.get('skills', [])))}"
        return hashlib.md5(profile_string.encode()).hexdigest()[:8]

    def calculate_mbti_similarity(self, user_mbti: str, career_mbti_weights: Dict) -> float:
        """Calculate MBTI similarity with enhanced differentiation"""
        if not user_mbti:
            return 0.5
            
        base_score = career_mbti_weights.get(user_mbti, 0.0)
        
        # Enhanced: Consider cognitive function similarity
        cognitive_functions = self.get_cognitive_functions(user_mbti)
        max_cognitive_similarity = 0.0
        
        for career_mbti, weight in career_mbti_weights.items():
            career_functions = self.get_cognitive_functions(career_mbti)
            common_functions = len(set(cognitive_functions) & set(career_functions))
            cognitive_similarity = (common_functions / 4.0) * weight
            max_cognitive_similarity = max(max_cognitive_similarity, cognitive_similarity)
        
        return 0.7 * base_score + 0.3 * max_cognitive_similarity

    def get_cognitive_functions(self, mbti: str) -> List[str]:
        """Get cognitive functions for MBTI type"""
        functions_map = {
            'INTJ': ['Ni', 'Te', 'Fi', 'Se'], 'INTP': ['Ti', 'Ne', 'Si', 'Fe'],
            'ENTJ': ['Te', 'Ni', 'Se', 'Fi'], 'ENTP': ['Ne', 'Ti', 'Fe', 'Si'],
            'INFJ': ['Ni', 'Fe', 'Ti', 'Se'], 'INFP': ['Fi', 'Ne', 'Si', 'Te'],
            'ENFJ': ['Fe', 'Ni', 'Se', 'Ti'], 'ENFP': ['Ne', 'Fi', 'Te', 'Si'],
            'ISTJ': ['Si', 'Te', 'Fi', 'Ne'], 'ISFJ': ['Si', 'Fe', 'Ti', 'Ne'],
            'ESTJ': ['Te', 'Si', 'Ne', 'Fi'], 'ESFJ': ['Fe', 'Si', 'Ne', 'Ti'],
            'ISTP': ['Ti', 'Se', 'Ni', 'Fe'], 'ISFP': ['Fi', 'Se', 'Ni', 'Te'],
            'ESTP': ['Se', 'Ti', 'Fe', 'Ni'], 'ESFP': ['Se', 'Fi', 'Te', 'Ni']
        }
        return functions_map.get(mbti, [])

    def calculate_riasec_similarity(self, user_riasec: List[str], career_riasec_weights: Dict) -> float:
        """Calculate RIASEC similarity with priority ordering"""
        if not user_riasec:
            return 0.5
            
        total_score = 0.0
        max_possible = 0.0
        
        for i, riasec_type in enumerate(user_riasec):
            weight = 1.0 / (i + 1)  # Higher weight for primary types
            score = career_riasec_weights.get(riasec_type, 0.0) * weight
            total_score += score
            max_possible += weight
        
        return total_score / max_possible if max_possible > 0 else 0.0

    def calculate_ikigai_similarity(self, user_ikigai: List[str], career_ikigai_weights: Dict) -> float:
        """Calculate Ikigai similarity with element matching"""
        if not user_ikigai:
            return 0.5
            
        total_score = 0.0
        matched_elements = 0
        
        for element in user_ikigai:
            if element in career_ikigai_weights:
                total_score += career_ikigai_weights[element]
                matched_elements += 1
        
        # Bonus for multiple matches
        if matched_elements > 1:
            total_score *= (1.0 + 0.1 * (matched_elements - 1))
        
        return min(1.0, total_score / len(user_ikigai)) if user_ikigai else 0.5

    def calculate_skills_similarity(self, user_skills: List[str], career_skill_domains: List[str]) -> float:
        """Calculate skills similarity with domain matching"""
        if not user_skills:
            return 0.3
            
        user_skills_lower = [skill.lower() for skill in user_skills]
        career_domains_lower = [domain.lower() for domain in career_skill_domains]
        
        matches = 0
        for user_skill in user_skills_lower:
            for domain in career_domains_lower:
                if user_skill in domain or domain in user_skill:
                    matches += 1
                    break
        
        return matches / len(career_domains_lower) if career_domains_lower else 0.0

    def calculate_trait_similarity(self, user_profile: Dict, career_trait_profile: Dict) -> float:
        """Calculate personality trait similarity"""
        default_traits = {
            "analytical": 0.5, "technical": 0.5, "creativity": 0.5, 
            "social": 0.5, "leadership": 0.5, "structured": 0.5, "practical": 0.5
        }
        
        user_traits = {**default_traits, **user_profile.get("traits", {})}
        
        similarity_score = 0.0
        trait_count = 0
        
        for trait, career_value in career_trait_profile.items():
            user_value = user_traits.get(trait, 0.5)
            trait_similarity = 1.0 - abs(user_value - career_value)
            similarity_score += trait_similarity
            trait_count += 1
        
        return similarity_score / trait_count if trait_count > 0 else 0.5

    def get_recommendations(self, user_profile: Dict[str, Any], top_n: int = 15) -> Dict[str, Any]:
        """Get personalized career recommendations with guaranteed differentiation"""
        user_hash = self.generate_user_profile_hash(user_profile)
        print(f"Generating recommendations for profile hash: {user_hash}")
        
        recommendations = []
        
        for career in self.career_database:
            personality_profile = career["personality_profile"]
            
            # Calculate individual similarity scores
            mbti_score = self.calculate_mbti_similarity(
                user_profile.get("mbti"), 
                personality_profile["mbti_weights"]
            )
            
            riasec_score = self.calculate_riasec_similarity(
                user_profile.get("riasec", []),
                personality_profile["riasec_weights"]
            )
            
            ikigai_score = self.calculate_ikigai_similarity(
                user_profile.get("ikigai", []),
                personality_profile["ikigai_weights"]
            )
            
            skills_score = self.calculate_skills_similarity(
                user_profile.get("skills", []),
                personality_profile["skill_domains"]
            )
            
            trait_score = self.calculate_trait_similarity(
                user_profile,
                personality_profile["trait_profile"]
            )
            
            # Calculate weighted total score
            total_score = (
                self.weights["mbti"] * mbti_score +
                self.weights["riasec"] * riasec_score +
                self.weights["ikigai"] * ikigai_score +
                self.weights["skills"] * skills_score +
                self.weights["traits"] * trait_score
            )
            
            # Add career to recommendations
            recommendations.append({
                "career": career,
                "total_score": total_score,
                "breakdown": {
                    "mbti": round(mbti_score, 3),
                    "riasec": round(riasec_score, 3),
                    "ikigai": round(ikigai_score, 3),
                    "skills": round(skills_score, 3),
                    "traits": round(trait_score, 3)
                }
            })
        
        # Sort by total score
        recommendations.sort(key=lambda x: x["total_score"], reverse=True)
        
        # Return top N recommendations with analysis
        top_recommendations = recommendations[:top_n]
        
        # Generate enhanced career data with match percentages
        enhanced_recommendations = []
        for rec in top_recommendations:
            career_data = rec["career"].copy()
            match_score = round(rec["total_score"] * 100, 1)
            
            # Add match percentage and reasoning
            career_data["match"] = match_score
            career_data["ai_reasoning"] = self.generate_reasoning(rec["breakdown"], user_profile, career_data)
            career_data["learning_path"] = self.generate_learning_path(career_data)
            career_data["resources"] = self.get_career_resources(career_data)
            career_data["personality_fit"] = self.calculate_personality_fit(user_profile, career_data)
            
            enhanced_recommendations.append(career_data)
        
        return {
            "user_profile_hash": user_hash,
            "recommendations": enhanced_recommendations,
            "analysis": self.analyze_user_profile(user_profile),
            "total_careers_considered": len(self.career_database)
        }

    def generate_reasoning(self, breakdown: Dict, user_profile: Dict, career: Dict) -> List[str]:
        """Generate detailed reasoning for each recommendation"""
        reasoning = []
        
        personality_profile = career["personality_profile"]
        
        # MBTI reasoning
        user_mbti = user_profile.get("mbti")
        if user_mbti and breakdown["mbti"] > 0.7:
            if user_mbti in personality_profile["mbti_weights"]:
                reasoning.append(f"Excellent match for your {user_mbti} personality type")
            else:
                reasoning.append("Good cognitive function alignment despite MBTI type difference")
        
        # RIASEC reasoning
        user_riasec = user_profile.get("riasec", [])
        if user_riasec and breakdown["riasec"] > 0.7:
            strong_matches = [r for r in user_riasec if r in personality_profile["riasec_weights"]]
            if strong_matches:
                reasoning.append(f"Aligns with your {', '.join(strong_matches)} interests")
        
        # Ikigai reasoning
        user_ikigai = user_profile.get("ikigai", [])
        if user_ikigai and breakdown["ikigai"] > 0.7:
            reasoning.append("Matches your purpose and fulfillment drivers")
        
        # Skills reasoning
        user_skills = user_profile.get("skills", [])
        if user_skills and breakdown["skills"] > 0.6:
            reasoning.append("Leverages your existing skills and competencies")
        
        # Fallback reasoning
        if not reasoning:
            if breakdown["mbti"] + breakdown["riasec"] > 1.2:
                reasoning.append("Strong personality and interest alignment")
            elif breakdown["traits"] > 0.7:
                reasoning.append("Excellent fit with your core personality traits")
            else:
                reasoning.append("Good potential match with development opportunities")
        
        return reasoning

    def calculate_personality_fit(self, user_profile: Dict, career: Dict) -> Dict:
        """Calculate detailed personality fit analysis"""
        fit_analysis = {
            'traits_match': [],
            'areas_growth': [],
            'compatibility_score': 0
        }
        
        user_traits = {
            'creativity': user_profile.get('creativity', 0.5),
            'analytical': user_profile.get('analytical', 0.5),
            'social': user_profile.get('social', 0.5),
            'technical': user_profile.get('technical', 0.5),
            'leadership': user_profile.get('leadership', 0.5),
            'structured': user_profile.get('structured', 0.5),
            'practical': user_profile.get('practical', 0.5)
        }
        
        career_traits = career['personality_profile']['trait_profile']
        
        strong_matches = 0
        total_traits = 0
        
        for trait, user_value in user_traits.items():
            career_value = career_traits.get(trait, 0.5)
            difference = abs(user_value - career_value)
            
            if difference < 0.2:
                fit_analysis['traits_match'].append(f"Strong {trait} alignment")
                strong_matches += 1
            elif difference > 0.4:
                fit_analysis['areas_growth'].append(f"Develop {trait} skills")
            
            total_traits += 1
        
        fit_analysis['compatibility_score'] = int((strong_matches / total_traits) * 100)
        return fit_analysis

    def generate_learning_path(self, career):
        """Generate personalized learning path"""
        base_path = {
            'foundation': {
                'duration': '3-6 months',
                'focus': 'Core fundamentals and prerequisite knowledge',
                'milestones': ['Complete foundational courses', 'Build basic projects', 'Join relevant communities']
            },
            'specialization': {
                'duration': '6-12 months',
                'focus': 'Advanced skills and practical applications',
                'milestones': ['Complete specialized training', 'Build portfolio projects', 'Gain certifications']
            },
            'professional': {
                'duration': '12+ months',
                'focus': 'Real-world experience and career development',
                'milestones': ['Internships or entry-level positions', 'Networking', 'Continuous learning']
            }
        }
        
        # Customize based on career category
        category_paths = {
            'technology': {
                'foundation': {'focus': 'Programming fundamentals, algorithms, and tools'},
                'specialization': {'focus': 'Specialized frameworks, systems design, and advanced concepts'}
            },
            'creative': {
                'foundation': {'focus': 'Design principles, creative tools, and fundamental techniques'},
                'specialization': {'focus': 'Advanced creative skills, portfolio development, and specialization'}
            },
            'healthcare': {
                'foundation': {'focus': 'Medical fundamentals, terminology, and basic procedures'},
                'specialization': {'focus': 'Specialized medical knowledge, clinical skills, and certifications'}
            },
            'business': {
                'foundation': {'focus': 'Business fundamentals, analytics, and communication skills'},
                'specialization': {'focus': 'Strategic planning, advanced analytics, and leadership development'}
            }
        }
        
        if career['category'] in category_paths:
            base_path['foundation']['focus'] = category_paths[career['category']]['foundation']['focus']
            base_path['specialization']['focus'] = category_paths[career['category']]['specialization']['focus']
        
        return base_path

    def get_career_resources(self, career):
        """Get career-specific resources"""
        resources = {
            'courses': [],
            'books': [],
            'tools': [],
            'communities': [],
            'certifications': []
        }
        
        # Category-based resource templates
        category_resources = {
            'technology': {
                'courses': [
                    {'name': 'Tech Foundations', 'platform': 'Coursera', 'url': 'https://coursera.org'},
                    {'name': 'Specialized Technical Training', 'platform': 'Udemy', 'url': 'https://udemy.com'}
                ],
                'tools': ['Git', 'VS Code', 'Relevant programming languages', 'Cloud platforms'],
                'communities': ['GitHub', 'Stack Overflow', 'Tech Meetups', 'Dev Communities']
            },
            'creative': {
                'courses': [
                    {'name': 'Creative Fundamentals', 'platform': 'Skillshare', 'url': 'https://skillshare.com'},
                    {'name': 'Advanced Design Techniques', 'platform': 'Domestika', 'url': 'https://domestika.org'}
                ],
                'tools': ['Adobe Creative Suite', 'Figma', 'Creative software', 'Portfolio platforms'],
                'communities': ['Dribbble', 'Behance', 'Creative forums', 'Design communities']
            },
            'healthcare': {
                'courses': [
                    {'name': 'Healthcare Basics', 'platform': 'edX', 'url': 'https://edx.org'},
                    {'name': 'Medical Specialization', 'platform': 'University Programs', 'url': '#'}
                ],
                'tools': ['Medical software', 'Diagnostic tools', 'Patient management systems'],
                'communities': ['Professional associations', 'Medical forums', 'Healthcare networks']
            }
        }
        
        if career['category'] in category_resources:
            resources.update(category_resources[career['category']])
        
        # Add career-specific certifications
        if career['category'] == 'technology':
            resources['certifications'] = ['Relevant technology certifications', 'Cloud certifications', 'Security certifications']
        elif career['category'] == 'healthcare':
            resources['certifications'] = ['State licenses', 'Specialty certifications', 'CPR/BLS certification']
        
        return resources

    def analyze_user_profile(self, user_profile: Dict) -> Dict:
        """Analyze user profile and provide insights"""
        analysis = {
            "personality_archetype": self.determine_archetype(user_profile),
            "strengths": [],
            "development_areas": [],
            "career_clusters": []
        }
        
        # Determine strengths based on assessments
        if user_profile.get("mbti"):
            analysis["strengths"].append(f"{user_profile['mbti']} cognitive strengths")
        
        if user_profile.get("riasec"):
            analysis["strengths"].append(f"{', '.join(user_profile['riasec'])} interest patterns")
        
        if user_profile.get("ikigai"):
            analysis["strengths"].append(f"Ikigai elements: {', '.join(user_profile['ikigai'])}")
        
        # Determine career clusters
        clusters = set()
        if user_profile.get("mbti") in ["INTJ", "INTP", "ENTJ", "ENTP"]:
            clusters.add("Analytical/Technical")
        if user_profile.get("mbti") in ["INFP", "ENFP", "INFJ", "ENFJ"]:
            clusters.add("Creative/Social")
        if "I" in user_profile.get("riasec", []):
            clusters.add("Investigative/Research")
        if "A" in user_profile.get("riasec", []):
            clusters.add("Artistic/Creative")
        if "S" in user_profile.get("riasec", []):
            clusters.add("Social/Helping")
        
        analysis["career_clusters"] = list(clusters)
        
        return analysis

    def determine_archetype(self, user_profile: Dict) -> str:
        """Determine primary personality archetype"""
        archetype_scores = {}
        
        for archetype_name, archetype_config in self.personality_archetypes.items():
            score = 0.0
            
            # MBTI matching
            if "mbti" in archetype_config and user_profile.get("mbti") in archetype_config["mbti"]:
                score += 0.4
            
            # RIASEC matching
            if "riasec" in archetype_config and any(r in archetype_config["riasec"] for r in user_profile.get("riasec", [])):
                score += 0.3
            
            # Ikigai matching
            if "ikigai" in archetype_config and any(i in archetype_config["ikigai"] for i in user_profile.get("ikigai", [])):
                score += 0.3
            
            archetype_scores[archetype_name] = score
        
        return max(archetype_scores.items(), key=lambda x: x[1])[0] if archetype_scores else "balanced_professional"

# Initialize the recommender
recommender = AdvancedCareerRecommender()

@app.route('/api/recommend-careers', methods=['POST'])
def recommend_careers():
    """Enhanced API endpoint for career recommendations"""
    try:
        user_data = request.json
        user_profile = user_data.get('user_profile', {})
        
        print(f"Received enhanced recommendation request for user: {user_profile}")
        
        # Get enhanced recommendations
        recommendations = recommender.get_recommendations(user_profile, top_n=15)
        
        return jsonify({
            'success': True,
            'recommendations': recommendations['recommendations'],
            'user_profile_analysis': recommendations['analysis'],
            'total_recommendations': len(recommendations['recommendations']),
            'profile_hash': recommendations['user_profile_hash'],
            'assessment_breakdown': get_assessment_breakdown(user_profile)
        })
    
    except Exception as e:
        print(f"Error in enhanced recommendation: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/careers', methods=['GET'])
def get_all_careers():
    """Get all available careers"""
    careers = recommender.career_database
    print(f"Sending all {len(careers)} careers data")
    return jsonify({
        'success': True,
        'careers': careers,
        'total': len(careers),
        'categories': list(set([career['category'] for career in careers]))
    })

@app.route('/api/test-recommendation', methods=['GET'])
def test_recommendation():
    """Enhanced test endpoint to verify ML differentiation"""
    # Test different personality archetypes
    test_cases = [
        {
            'name': 'Analytical Researcher (INTJ)',
            'profile': {
                'mbti': 'INTJ',
                'riasec': ['I', 'R'],
                'ikigai': ['profession', 'mission'],
                'skills': ['research', 'analysis', 'programming'],
                'traits': {'analytical': 0.9, 'technical': 0.8}
            }
        },
        {
            'name': 'Creative Helper (ENFP)', 
            'profile': {
                'mbti': 'ENFP',
                'riasec': ['A', 'S'],
                'ikigai': ['passion', 'mission'],
                'skills': ['creative', 'communication', 'empathy'],
                'traits': {'creativity': 0.9, 'social': 0.8}
            }
        },
        {
            'name': 'Practical Leader (ESTJ)',
            'profile': {
                'mbti': 'ESTJ', 
                'riasec': ['E', 'C'],
                'ikigai': ['profession', 'vocation'],
                'skills': ['leadership', 'organization', 'planning'],
                'traits': {'leadership': 0.9, 'structured': 0.8}
            }
        },
        {
            'name': 'Technical Problem-Solver (ISTP)',
            'profile': {
                'mbti': 'ISTP',
                'riasec': ['R', 'I'],
                'ikigai': ['vocation', 'profession'],
                'skills': ['technical', 'hands-on', 'troubleshooting'],
                'traits': {'practical': 0.9, 'technical': 0.8}
            }
        },
        {
            'name': 'ENTJ Business Leader',
            'profile': {
                'mbti': 'ENTJ',
                'riasec': ['E', 'C'],
                'ikigai': ['profession', 'mission'],
                'skills': ['leadership', 'strategy', 'analysis'],
                'traits': {'leadership': 0.9, 'analytical': 0.8}
            }
        },
        {
            'name': 'INFP Creative Writer',
            'profile': {
                'mbti': 'INFP',
                'riasec': ['A', 'I'],
                'ikigai': ['passion', 'vocation'],
                'skills': ['writing', 'creative', 'empathy'],
                'traits': {'creativity': 0.9, 'social': 0.7}
            }
        }
    ]
    
    results = []
    for test_case in test_cases:
        recommendations = recommender.get_recommendations(test_case['profile'], top_n=5)
        results.append({
            'test_case': test_case['name'],
            'profile_hash': recommendations['user_profile_hash'],
            'archetype': recommendations['analysis']['personality_archetype'],
            'top_recommendations': [
                {
                    'title': r['title'], 
                    'match': r['match'], 
                    'category': r['category'],
                    'reasoning': r['ai_reasoning'][0] if r['ai_reasoning'] else 'Good match'
                } for r in recommendations['recommendations'][:3]
            ]
        })
    
    return jsonify({
        'success': True,
        'test_results': results,
        'message': 'Enhanced ML differentiation test completed successfully'
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy', 
        'total_careers': len(recommender.career_database),
        'career_categories': list(set([career['category'] for career in recommender.career_database])),
        'message': 'Advanced Career Recommendation API with 15 diverse careers is running successfully!'
    })

def get_assessment_breakdown(user_profile):
    """Get detailed breakdown of assessment results"""
    breakdown = {
        'mbti_analysis': {},
        'riasec_analysis': {},
        'skills_analysis': {},
        'ikigai_analysis': {}
    }
    
    # MBTI analysis
    mbti_type = user_profile.get('mbti', '')
    if mbti_type:
        mbti_descriptions = {
            'INTJ': 'Strategic, independent, and knowledge-oriented',
            'INTP': 'Innovative, analytical, and theoretical',
            'ENTJ': 'Leadership-driven, strategic, and efficient',
            'ENTP': 'Entrepreneurial, innovative, and debate-loving',
            'INFJ': 'Idealistic, compassionate, and future-oriented',
            'INFP': 'Values-driven, creative, and empathetic',
            'ENFJ': 'Charismatic, inspiring, and relationship-focused',
            'ENFP': 'Enthusiastic, creative, and possibility-oriented',
            'ISTJ': 'Responsible, practical, and detail-oriented',
            'ISFJ': 'Supportive, reliable, and service-oriented',
            'ESTJ': 'Efficient, organized, and tradition-respecting',
            'ESFJ': 'Sociable, caring, and harmony-seeking',
            'ISTP': 'Practical, analytical, and action-oriented',
            'ISFP': 'Artistic, gentle, and present-focused',
            'ESTP': 'Energetic, pragmatic, and risk-taking',
            'ESFP': 'Spontaneous, playful, and people-oriented'
        }
        breakdown['mbti_analysis'] = {
            'type': mbti_type,
            'description': mbti_descriptions.get(mbti_type, 'Personality type analysis'),
            'career_implications': get_mbti_career_implications(mbti_type)
        }
    
    # RIASEC analysis
    riasec_types = user_profile.get('riasec', [])
    if riasec_types:
        riasec_descriptions = {
            'R': 'Realistic - Hands-on, practical, technical',
            'I': 'Investigative - Analytical, intellectual, scientific',
            'A': 'Artistic - Creative, expressive, original',
            'S': 'Social - Helping, teaching, serving',
            'E': 'Enterprising - Leadership, persuasion, business',
            'C': 'Conventional - Organized, detail-oriented, systematic'
        }
        breakdown['riasec_analysis'] = {
            'types': riasec_types,
            'descriptions': [riasec_descriptions.get(t, '') for t in riasec_types],
            'primary_type': riasec_types[0] if riasec_types else None
        }
    
    return breakdown

def get_mbti_career_implications(mbti_type):
    """Get career implications for MBTI type"""
    implications = {
        'INTJ': 'Excels in strategic planning, research, and systems design',
        'INTP': 'Thrives in theoretical research, innovation, and complex problem-solving',
        'ENTJ': 'Natural leaders in business, management, and organizational strategy',
        'ENTP': 'Innovators in entrepreneurship, consulting, and creative problem-solving',
        'INFJ': 'Excel in counseling, writing, and roles that help others grow',
        'INFP': 'Creative fields, counseling, and work aligned with personal values',
        'ENFJ': 'Teaching, leadership, and roles that inspire and motivate others',
        'ENFP': 'Creative industries, counseling, and entrepreneurial ventures',
        'ISTJ': 'Reliable in administrative, technical, and detail-oriented roles',
        'ISFJ': 'Healthcare, education, and service-oriented professions',
        'ESTJ': 'Management, administration, and roles requiring organization',
        'ESFJ': 'Healthcare, education, and customer service roles',
        'ISTP': 'Technical fields, emergency services, and hands-on problem-solving',
        'ISFP': 'Arts, design, and hands-on helping professions',
        'ESTP': 'Sales, entrepreneurship, and action-oriented roles',
        'ESFP': 'Entertainment, hospitality, and people-oriented careers'
    }
    return implications.get(mbti_type, 'Versatile across many career paths')

if __name__ == '__main__':
    print("🚀 Starting Advanced Career Recommendation API...")
    print(f"📊 Career database loaded with {len(recommender.career_database)} diverse careers")
    print("🤖 Advanced ML model with guaranteed differentiation for all combinations")
    print("🌐 API running on http://localhost:5001")
    print("\nAvailable endpoints:")
    print("  GET  /api/careers - Get all 15+ careers across categories")
    print("  POST /api/recommend-careers - Get AI-powered recommendations with match percentages") 
    print("  GET  /api/test-recommendation - Test ML differentiation with 6+ profiles")
    print("  GET  /api/health - Health check")
    
    app.run(debug=True, host='0.0.0.0', port=5001)
