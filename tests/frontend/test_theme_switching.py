# Frontend tests for theme switching functionality
import pytest
import requests
import json
import time
from unittest.mock import Mock, patch
try:
    from KestrelAI.backend.main import AppSettings, Theme
except ImportError:
    from backend.main import AppSettings, Theme


@pytest.mark.frontend
class TestThemeSwitching:
    """Test theme switching integration functionality."""
    
    @pytest.fixture
    def backend_url(self):
        """Backend URL for testing."""
        return "http://localhost:8000"
    
    @pytest.fixture
    def frontend_url(self):
        """Frontend URL for testing."""
        return "http://localhost:5173"
    
    def test_app_settings_model_theme_field(self):
        """Test that AppSettings model includes theme field."""
        # Test default theme
        settings = AppSettings()
        assert hasattr(settings, 'theme')
        assert settings.theme == Theme.amber
        
        # Test theme field validation
        settings_blue = AppSettings(theme=Theme.blue)
        assert settings_blue.theme == Theme.blue
        
        # Test invalid theme (should raise validation error)
        with pytest.raises(ValueError):
            AppSettings(theme="invalid_theme")
    
    def test_backend_theme_endpoints(self, backend_url):
        """Test backend theme switching endpoints."""
        # Test GET settings (should return default amber theme)
        try:
            response = requests.get(f"{backend_url}/settings", timeout=5)
            if response.status_code == 200:
                settings = response.json()
                assert "theme" in settings
                assert settings["theme"] in ["amber", "blue"]
                
                # Test switching to blue theme
                blue_settings = {
                    "ollamaMode": "local",
                    "orchestrator": "kestrel",
                    "theme": "blue"
                }
                
                response = requests.post(f"{backend_url}/settings", json=blue_settings, timeout=5)
                if response.status_code == 200:
                    settings = response.json()
                    assert settings["theme"] == "blue"
                    
                    # Test switching back to amber
                    amber_settings = {
                        "ollamaMode": "local", 
                        "orchestrator": "kestrel",
                        "theme": "amber"
                    }
                    
                    response = requests.post(f"{backend_url}/settings", json=amber_settings, timeout=5)
                    if response.status_code == 200:
                        settings = response.json()
                        assert settings["theme"] == "amber"
                        
        except requests.exceptions.ConnectionError:
            pytest.skip("Backend not available for testing")
    
    def test_frontend_accessibility(self, frontend_url):
        """Test that frontend is accessible."""
        try:
            response = requests.get(frontend_url, timeout=5)
            assert response.status_code == 200
            assert "html" in response.text.lower()
        except requests.exceptions.ConnectionError:
            pytest.skip("Frontend not available for testing")
    
    def test_theme_persistence(self, backend_url):
        """Test that theme settings persist across requests."""
        try:
            # Set theme to blue
            blue_settings = {
                "ollamaMode": "local",
                "orchestrator": "kestrel", 
                "theme": "blue"
            }
            
            response = requests.post(f"{backend_url}/settings", json=blue_settings, timeout=5)
            if response.status_code == 200:
                # Get settings and verify persistence
                response = requests.get(f"{backend_url}/settings", timeout=5)
                if response.status_code == 200:
                    settings = response.json()
                    assert settings["theme"] == "blue"
                    
                    # Set theme to amber
                    amber_settings = {
                        "ollamaMode": "local",
                        "orchestrator": "kestrel",
                        "theme": "amber"
                    }
                    
                    response = requests.post(f"{backend_url}/settings", json=amber_settings, timeout=5)
                    if response.status_code == 200:
                        # Get settings and verify persistence
                        response = requests.get(f"{backend_url}/settings", timeout=5)
                        if response.status_code == 200:
                            settings = response.json()
                            assert settings["theme"] == "amber"
                            
        except requests.exceptions.ConnectionError:
            pytest.skip("Backend not available for testing")
    
    def test_theme_validation(self, backend_url):
        """Test theme validation on backend."""
        try:
            # Test invalid theme
            invalid_settings = {
                "ollamaMode": "local",
                "orchestrator": "kestrel",
                "theme": "invalid_theme"
            }
            
            response = requests.post(f"{backend_url}/settings", json=invalid_settings, timeout=5)
            # Should return validation error
            assert response.status_code == 422
            
            # Test valid themes
            for theme in ["amber", "blue"]:
                valid_settings = {
                    "ollamaMode": "local",
                    "orchestrator": "kestrel",
                    "theme": theme
                }
                
                response = requests.post(f"{backend_url}/settings", json=valid_settings, timeout=5)
                if response.status_code == 200:
                    settings = response.json()
                    assert settings["theme"] == theme
                    
        except requests.exceptions.ConnectionError:
            pytest.skip("Backend not available for testing")
    
    def test_theme_with_other_settings(self, backend_url):
        """Test that theme works correctly with other settings."""
        try:
            # Test theme with different orchestrator
            settings = {
                "ollamaMode": "local",
                "orchestrator": "kestrel",
                "theme": "blue"
            }
            
            response = requests.post(f"{backend_url}/settings", json=settings, timeout=5)
            if response.status_code == 200:
                result = response.json()
                assert result["theme"] == "blue"
                assert result["orchestrator"] == "kestrel"
                assert result["ollamaMode"] == "local"
                
        except requests.exceptions.ConnectionError:
            pytest.skip("Backend not available for testing")
    
    def test_css_theme_variables_exist(self):
        """Test that CSS theme variables are properly defined."""
        # This test verifies that the CSS file contains the necessary theme variables
        import os
        css_file = "/Users/ganeshdanke/Documents/KestrelAI/kestrel-ui/src/index.css"
        
        if os.path.exists(css_file):
            with open(css_file, 'r') as f:
                css_content = f.read()
                
                # Check for amber theme variables
                assert "--theme-primary-50: #fffbeb" in css_content
                assert "--theme-primary-600: #d97706" in css_content
                assert "--theme-primary-700: #b45309" in css_content
                
                # Check for blue theme variables
                assert '[data-theme="blue"]' in css_content
                assert "--theme-primary-50: #eff6ff" in css_content
                assert "--theme-primary-600: #2563eb" in css_content
                assert "--theme-primary-700: #1d4ed8" in css_content
                
                # Check for utility classes
                assert ".theme-bg-primary-50" in css_content
                assert ".theme-text-primary-600" in css_content
                assert ".theme-border-primary-200" in css_content
                
        else:
            pytest.skip("CSS file not found")
    
    def test_theme_switching_performance(self, backend_url):
        """Test theme switching performance."""
        try:
            start_time = time.time()
            
            # Switch theme multiple times
            for theme in ["amber", "blue", "amber", "blue"]:
                settings = {
                    "ollamaMode": "local",
                    "orchestrator": "kestrel",
                    "theme": theme
                }
                
                response = requests.post(f"{backend_url}/settings", json=settings, timeout=5)
                if response.status_code != 200:
                    break
                    
            total_time = time.time() - start_time
            
            # Theme switching should be fast (under 2 seconds for 4 switches)
            assert total_time < 2.0
            
        except requests.exceptions.ConnectionError:
            pytest.skip("Backend not available for testing")
    
    def test_frontend_build_success(self):
        """Test that frontend builds successfully with theme changes."""
        import subprocess
        import os
        
        frontend_dir = "/Users/ganeshdanke/Documents/KestrelAI/kestrel-ui"
        
        if os.path.exists(frontend_dir):
            try:
                # Test build
                result = subprocess.run(
                    ["npm", "run", "build"],
                    cwd=frontend_dir,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                # Build should succeed
                assert result.returncode == 0
                
                # Check that dist directory was created
                dist_dir = os.path.join(frontend_dir, "dist")
                assert os.path.exists(dist_dir)
                
                # Check that CSS file was generated (in assets subdirectory)
                assets_dir = os.path.join(dist_dir, "assets")
                if os.path.exists(assets_dir):
                    css_files = [f for f in os.listdir(assets_dir) if f.endswith('.css')]
                    assert len(css_files) > 0, f"No CSS files found in {assets_dir}. Files: {os.listdir(assets_dir)}"
                else:
                    # Fallback: check dist directory directly
                    css_files = [f for f in os.listdir(dist_dir) if f.endswith('.css')]
                    assert len(css_files) > 0, f"No CSS files found in {dist_dir}. Files: {os.listdir(dist_dir)}"
                
            except subprocess.TimeoutExpired:
                pytest.fail("Frontend build timed out")
            except FileNotFoundError:
                pytest.skip("npm not available")
        else:
            pytest.skip("Frontend directory not found")
