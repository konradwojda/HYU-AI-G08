import React, { useState } from "react";
import axios from "axios";
import Card from "react-bootstrap/Card";
import ListGroup from "react-bootstrap/ListGroup";
import Button from "react-bootstrap/Button";

const MainPage = () => {
  const [imageSrc, setImageSrc] = useState(null); // For displaying the uploaded image
  const [analysisResults, setAnalysisResults] = useState([]); // For storing backend results
  const [loading, setLoading] = useState(false); // For showing loading state
  const [error, setError] = useState(null); // For handling errors

  // Handle file input change
  const handleFileChange = async (event) => {
    const file = event.target.files[0];
    if (file) {
      setImageSrc(URL.createObjectURL(file)); // Show the uploaded image locally
      setLoading(true);
      setError(null);

      const formData = new FormData();
      formData.append("file", file);

      try {
        // Send file to backend
        const response = await axios.post("http://localhost:5000/upload", formData, {
          headers: { "Content-Type": "multipart/form-data" },
        });

        // Update results from backend response
        setAnalysisResults([response.data.message]);
      } catch (err) {
        if (err.response) {
            // Backend responded with an error
            console.error("Backend error:", err.response.data);
            setError(err.response.data.error || "An error occurred on the server.");
          } else if (err.request) {
            // No response received from backend
            console.error("No response from server:", err.request);
            setError("No response from server.");
          } else {
            // Axios configuration error
            console.error("Error during request setup:", err.message);
            setError("An error occurred while setting up the request.");
          }
      } finally {
        setLoading(false);
      }
    }
  };

  return (
 
    <div className="imageAnalysis">
      {!imageSrc && (
        <div>
          <h2>Upload an Image for Analysis</h2>
          <input type="file" accept="image/*" onChange={handleFileChange} />
        </div>
      )}
      {imageSrc && (
        <Card className="card-container">
          <Card.Img variant="top" src={imageSrc} alt="Uploaded Image" className="card-img" />
          <Card.Body>
            <Card.Title>Image Results</Card.Title>
            {loading ? (
              <Card.Text>Analyzing the image, please wait...</Card.Text>
            ) : error ? (
              <Card.Text className="text-danger">{error}</Card.Text>
            ) : (
              <Card.Text>
                Below are the results of the analysis for the uploaded image:
              </Card.Text>
            )}
          </Card.Body>
          <ListGroup>
            {loading
              ? null
              : analysisResults.map((result, index) => (
                  <ListGroup.Item key={index}>{result}</ListGroup.Item>
                ))}
          </ListGroup>
          <Card.Body>
            
            <Button
              onClick={() => {
                setImageSrc(null);
                setAnalysisResults([]);
                setError(null);
              }}
              variant="primary"
            >
              Analyze another image!
            </Button>
          </Card.Body>
        </Card>
      )}
    </div>
  );
};

export default MainPage;
