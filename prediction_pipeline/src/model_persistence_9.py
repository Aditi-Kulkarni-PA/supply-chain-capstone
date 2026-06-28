"""
Model Persistence Module

This module contains functions for saving and loading trained models.
"""

import os
import pickle
import json
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd


class ModelPersistence:
    """Class for saving and loading trained models."""
    
    DEFAULT_MODEL_DIR = "models"
    
    @staticmethod
    def save_model(
        model: Any,
        model_name: str,
        model_dir: str = DEFAULT_MODEL_DIR,
        metadata: Optional[Dict[str, Any]] = None,
        overwrite: bool = False
    ) -> str:
        """
        Save a trained model to disk using pickle.
        
        Parameters:
        model: The trained model object to save
        model_name: Name for the model file (without extension)
        model_dir: Directory to save the model (default: "model")
        metadata: Optional dictionary with metadata about the model
        overwrite: Whether to overwrite existing file (default: False)
        
        Returns:
        Path to the saved model file
        """
        # Create directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Create full file path
        model_path = os.path.join(model_dir, f"{model_name}.pkl")
        
        # Check if file exists and overwrite flag
        if os.path.exists(model_path) and not overwrite:
            raise FileExistsError(
                f"Model file '{model_path}' already exists. "
                f"Set overwrite=True to replace it."
            )
        
        # Save the model
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save metadata if provided
        if metadata is not None:
            metadata_path = os.path.join(model_dir, f"{model_name}_metadata.json")
            # Convert any non-serializable objects in metadata
            metadata_serializable = ModelPersistence._make_serializable(metadata)
            with open(metadata_path, 'w') as f:
                json.dump(metadata_serializable, f, indent=2, default=str)
        
        print(f"✓ Model saved to: {model_path}")
        if metadata:
            print(f"✓ Metadata saved to: {os.path.join(model_dir, f'{model_name}_metadata.json')}")
        
        return model_path
    
    @staticmethod
    def load_model(
        model_name: str,
        model_dir: str = DEFAULT_MODEL_DIR
    ) -> Any:
        """
        Load a saved model from disk.
        
        Parameters:
        model_name: Name of the model file (without extension)
        model_dir: Directory where the model is saved (default: "model")
        
        Returns:
        The loaded model object
        """
        model_path = os.path.join(model_dir, f"{model_name}.pkl")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' not found.")
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        print(f"✓ Model loaded from: {model_path}")
        return model
    
    @staticmethod
    def load_metadata(
        model_name: str,
        model_dir: str = DEFAULT_MODEL_DIR
    ) -> Dict[str, Any]:
        """
        Load metadata for a saved model.
        
        Parameters:
        model_name: Name of the model file (without extension)
        model_dir: Directory where the model is saved (default: "model")
        
        Returns:
        Dictionary with model metadata
        """
        metadata_path = os.path.join(model_dir, f"{model_name}_metadata.json")
        
        if not os.path.exists(metadata_path):
            print(f"⚠ No metadata file found at: {metadata_path}")
            return {}
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"✓ Metadata loaded from: {metadata_path}")
        return metadata
    
    @staticmethod
    def save_shortlisted_models(
        regression_model: Optional[Any] = None,
        regression_name: Optional[str] = None,
        regression_metrics: Optional[Dict[str, Any]] = None,
        classification_model: Optional[Any] = None,
        classification_name: Optional[str] = None,
        classification_metrics: Optional[Dict[str, Any]] = None,
        model_dir: str = DEFAULT_MODEL_DIR,
        overwrite: bool = False
    ) -> Dict[str, str]:
        """
        Save shortlisted regression and/or classification models.
        
        Parameters:
        regression_model: The trained regression model to save
        regression_name: Name for the regression model file
        regression_metrics: Optional metrics dictionary for regression model
        classification_model: The trained classification model to save
        classification_name: Name for the classification model file
        classification_metrics: Optional metrics dictionary for classification model
        model_dir: Directory to save models (default: "model")
        overwrite: Whether to overwrite existing files (default: False)
        
        Returns:
        Dictionary with paths to saved models
        """
        saved_paths = {}
        
        # Save regression model if provided
        if regression_model is not None:
            if regression_name is None:
                regression_name = "best_regression_model"
            
            # Prepare metadata
            reg_metadata = {
                "model_type": "regression",
                "saved_at": datetime.now().isoformat(),
                "metrics": regression_metrics or {}
            }
            
            reg_path = ModelPersistence.save_model(
                regression_model,
                regression_name,
                model_dir=model_dir,
                metadata=reg_metadata,
                overwrite=overwrite
            )
            saved_paths["regression"] = reg_path
        
        # Save classification model if provided
        if classification_model is not None:
            if classification_name is None:
                classification_name = "best_classification_model"
            
            # Prepare metadata
            clf_metadata = {
                "model_type": "classification",
                "saved_at": datetime.now().isoformat(),
                "metrics": classification_metrics or {}
            }
            
            clf_path = ModelPersistence.save_model(
                classification_model,
                classification_name,
                model_dir=model_dir,
                metadata=clf_metadata,
                overwrite=overwrite
            )
            saved_paths["classification"] = clf_path
        
        if not saved_paths:
            print("⚠ No models provided to save.")
        
        return saved_paths
    
    @staticmethod
    def _make_serializable(obj: Any) -> Any:
        """
        Convert an object to a JSON-serializable format.
        
        Parameters:
        obj: Object to convert
        
        Returns:
        JSON-serializable version of the object
        """
        if isinstance(obj, dict):
            return {k: ModelPersistence._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [ModelPersistence._make_serializable(item) for item in obj]
        elif isinstance(obj, (pd.Series, pd.DataFrame)):
            return obj.to_dict()
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            # For other types, convert to string
            return str(obj)
    
    @staticmethod
    def list_saved_models(model_dir: str = DEFAULT_MODEL_DIR) -> Dict[str, list]:
        """
        List all saved models in the model directory.
        
        Parameters:
        model_dir: Directory to check (default: "model")
        
        Returns:
        Dictionary with lists of regression and classification models
        """
        if not os.path.exists(model_dir):
            print(f"⚠ Model directory '{model_dir}' does not exist.")
            return {"regression": [], "classification": []}
        
        models = {"regression": [], "classification": []}
        
        for filename in os.listdir(model_dir):
            if filename.endswith('.pkl'):
                model_name = filename[:-4]  # Remove .pkl extension
                
                # Try to load metadata to determine type
                try:
                    metadata = ModelPersistence.load_metadata(model_name, model_dir)
                    model_type = metadata.get("model_type", "unknown")
                    if model_type in models:
                        models[model_type].append(model_name)
                except:
                    models["regression"].append(model_name)  # Default to regression if unknown
        
        return models

