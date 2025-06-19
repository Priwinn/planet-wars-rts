from py4j.java_gateway import JavaGateway
import numpy as np
from typing import Dict, Any, List, Tuple
import subprocess
import time
import atexit
import os

class KotlinGameRunnerBridge:
    def __init__(self, jar_path: str = None, auto_start_kotlin: bool = True):
        self.gateway = None
        self.kotlin_process = None
        self.jar_path = jar_path or self._find_jar_path()
        
        if auto_start_kotlin:
            self.start_kotlin_gateway()
        
        atexit.register(self.cleanup)
    
    def _find_jar_path(self) -> str:
        """Find the built JAR file"""
        workspace_root = r"c:\Users\Ruizhe\Desktop\workspace\planet-wars-rts"
        jar_path = os.path.join(workspace_root, "app", "build", "libs", "py4J-gateway-server.jar")
        return jar_path
    
    def start_kotlin_gateway(self):
        """Start the Kotlin Py4J gateway server"""
        if self.kotlin_process is None:
            # Start the Kotlin gateway server
            cmd = [
                "java", "-jar", self.jar_path
            ]
            self.kotlin_process = subprocess.Popen(cmd)
            
            # Wait a moment for the server to start
            time.sleep(2)
        
        # Connect to the gateway
        self.gateway = JavaGateway()
        self.entry_point = self.gateway.entry_point
        self.game_runner = self.entry_point.createGameRunner()
    
    def new_game(self):
        """Start a new game"""
        self.game_runner.newGame()
    
    def step_game(self) -> Dict[str, Any]:
        """Step the game and return the state"""
        
        result = self.game_runner.stepGame()
        return self._convert_java_map(result)
    
    def get_game_state(self) -> Dict[str, Any]:
        """Get current game state"""
        result = self.game_runner.getGameState()
        return self._convert_java_map(result)
    
    def _convert_java_map(self, java_map) -> Dict[str, Any]:
        """Convert Java map to Python dict"""
        result = {}
        for key in java_map:
            value = java_map[key]
            if hasattr(value, '__iter__') and not isinstance(value, str):
                # Convert Java lists to Python lists
                result[key] = [self._convert_java_object(item) for item in value]
            else:
                result[key] = self._convert_java_object(value)
        return result
    
    def _convert_java_object(self, obj):
        """Convert Java objects to Python objects"""
        if hasattr(obj, '__iter__') and not isinstance(obj, str):
            if hasattr(obj, 'entrySet'):  # It's a map
                return {k: self._convert_java_object(v) for k, v in obj.items()}
            else:  # It's a list
                return [self._convert_java_object(item) for item in obj]
        return obj
    
    def cleanup(self):
        """Clean up resources"""
        if self.gateway:
            self.gateway.shutdown()
        if self.kotlin_process:
            self.kotlin_process.terminate()
            self.kotlin_process.wait()