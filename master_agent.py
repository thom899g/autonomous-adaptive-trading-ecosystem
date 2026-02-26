"""
MASTER AGENT - Autonomous Adaptive Trading Ecosystem
Core orchestrator managing sub-agents, strategy deployment, and system evolution.
Designed for high reliability with comprehensive error handling and logging.
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

# Firebase for state management (CRITICAL CONSTRAINT)
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    from firebase_admin.exceptions import FirebaseError
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    logging.warning("firebase-admin not available. State management will use local storage.")

# Standard libraries only - no hallucinations
import pandas as pd
import numpy as np
from typing import TypedDict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('trading_ecosystem.log')
    ]
)
logger = logging.getLogger(__name__)

class AgentStatus(Enum):
    """Agent lifecycle states"""
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    TERMINATED = "terminated"

class MarketType(Enum):
    """Supported market types"""
    CRYPTO = "crypto"
    FOREX = "forex"
    STOCKS = "stocks"
    FUTURES = "futures"

@dataclass
class AgentMetrics:
    """Performance metrics for agents"""
    uptime: float
    success_rate: float
    error_count: int
    last_active: datetime
    performance_score: float = 0.0

class MasterAgent:
    """Core orchestrator for the trading ecosystem"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize Master Agent with configuration"""
        self.status = AgentStatus.INITIALIZING
        self.sub_agents: Dict[str, Any] = {}
        self.active_strategies: List[str] = []
        self.metrics = AgentMetrics(
            uptime=0.0,
            success_rate=1.0,
            error_count=0,
            last_active=datetime.now()
        )
        
        # Initialize Firebase (CRITICAL CONSTRAINT)
        self.firebase_client = self._init_firebase()
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        
        logger.info("Master Agent initialized successfully")
    
    def _init_firebase(self) -> Optional[Any]:
        """Initialize Firebase connection with error handling"""
        if not FIREBASE_AVAILABLE:
            logger.warning("Firebase not available. Using local state management.")
            return None
            
        try:
            # Check for credentials file before initialization
            import os
            if os.path.exists("firebase-credentials.json"):
                cred = credentials.Certificate("firebase-credentials.json")
                firebase_admin.initialize_app(cred)
                client = firestore.client()
                logger.info("Firebase Firestore initialized successfully")
                return client
            else:
                logger.warning("Firebase credentials not found. Using local storage.")
                return None
        except Exception as e:
            logger.error(f"Firebase initialization failed: {str(e)}")
            return None
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration with validation"""
        default_config = {
            "polling_interval": 60,
            "max_concurrent_strategies": 10,
            "risk_tolerance": 0.02,
            "markets": [MarketType.CRYPTO.value],
            "logging_level": "INFO"
        }
        
        # In production, this would load from a file or database
        if config_path:
            try:
                import json
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    # Validate configuration
                    if not isinstance(loaded_config, dict):
                        raise ValueError("Configuration must be a dictionary")
                    
                    # Merge with defaults
                    default_config.update(loaded_config)
                    logger.info(f"Configuration loaded from {config_path}")
            except FileNotFoundError:
                logger.warning(f"Config file {config_path} not found. Using defaults.")
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in config file: {str(e)}")
            except Exception as e:
                logger.error(f"Error loading config: {str(e)}")
        
        return default_config
    
    def register_sub_agent(self, agent_id: str, agent_instance: Any) -> bool:
        """Register a sub-agent with validation"""
        if not agent_id or not isinstance(agent_id, str):
            logger.error("Invalid agent_id provided")
            return False
            
        if agent_id in self.sub_agents:
            logger.warning(f"Agent {agent_id} already registered. Updating.")
            
        self.sub_agents[agent_id] = {
            'instance': agent_instance,
            'status': AgentStatus.READY,
            'registered_at': datetime.now(),
            'last_heartbeat': datetime.now()
        }
        
        logger.info(f"Sub-agent {agent_id} registered successfully")
        return True
    
    async def start(self):
        """Start the Master Agent main loop"""
        self.status = AgentStatus.RUNNING
        logger.info("Master Agent starting main execution loop")
        
        try:
            # Start sub-agents
            await self._start_sub_agents()
            
            # Main execution loop
            while self.status == AgentStatus.RUNNING:
                try:
                    # Monitor sub-agents
                    await self._monitor_sub_agents()
                    
                    # Update strategies
                    await self._update_strategies()
                    
                    # Save state to Firebase
                    await self._save_state()
                    
                    # Update metrics
                    self._update_metrics()
                    
                    # Wait for next cycle
                    await asyncio.sleep(self.config.get("polling_interval", 60))
                    
                except asyncio.CancelledError:
                    logger.info("Main loop cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {str(e)}")
                    self.metrics.error_count += 1
                    await asyncio.sleep(5)  # Backoff on error
                    
        except Exception as e:
            logger.critical(f"Fatal error in Master Agent: {str(e)}")
            self.status = AgentStatus.ERROR
            raise
    
    async def _start_sub_agents(self):
        """Initialize and start all registered sub-agents"""
        for agent_id, agent_data in self.sub_agents.items():
            try:
                agent_instance = agent_data['instance']
                if hasattr(agent_instance, 'start'):
                    await agent_instance.start()
                    agent_data['status'] = AgentStatus.RUNNING
                    logger.info(f"Started sub-agent: {agent_id}")
                else:
                    logger.warning(f"Agent {agent_id} has no start method")
            except Exception as e:
                logger.error(f"Failed to start agent {agent_id}: {str(e)}")