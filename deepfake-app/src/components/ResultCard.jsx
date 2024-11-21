
// import Card from "react-bootstrap/Card"
// import ListGroup from 'react-bootstrap/ListGroup';
// import React, { useState } from "react";
// import axios from "axios";


// const ResultCard = ({src}) => {

//     return(
//         <div className="imageAnalysis">
//             <Card className="card-container">
//                 <Card.Img variant="top" src={src} alt="analysis" className="card-img"/>
//                 <Card.Body>
//                     <Card.Title>Image Results</Card.Title>
//                     <Card.Text> Lorem ipsum dolor sit amet, 
//                                 consectetur adipiscing elit, 
//                                 sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
//                     </Card.Text>
//                 </Card.Body>
//                 <ListGroup>
//                     <ListGroup.Item>Cras justo odio</ListGroup.Item>
//                     <ListGroup.Item>Cras justo odio</ListGroup.Item>
//                 </ListGroup>
//                 <Card.Body>
//                     Analyze another image!
//                 </Card.Body>
//             </Card>
//         </div>
//     );
// }

// export default ResultCard



import React from "react";
import Card from "react-bootstrap/Card";
import ListGroup from "react-bootstrap/ListGroup";
import Button from "react-bootstrap/Button";

const ResultCard = ({ src, analysisResults, onAnalyzeAnother, loading, error }) => {
  return (
    <Card className="mx-auto" style={{ maxWidth: "500px" }}>
      <Card.Img variant="top" src={src} alt="Uploaded Image" />
      <Card.Body>
        <Card.Title>Image Results</Card.Title>
        {loading ? (
          <Card.Text className="text-primary">Analyzing the image, please wait...</Card.Text>
        ) : error ? (
          <Card.Text className="text-danger">{error}</Card.Text>
        ) : (
          <Card.Text>Below are the results of the analysis:</Card.Text>
        )}
      </Card.Body>
      <ListGroup className="list-group-flush">
        {loading
          ? null
          : analysisResults.map((result, index) => (
              <ListGroup.Item key={index}>{result}</ListGroup.Item>
            ))}
      </ListGroup>
      <Card.Body className="text-center">
        <Button variant="primary" onClick={onAnalyzeAnother}>
          Analyze Another Image
        </Button>
      </Card.Body>
    </Card>
  );
};

export default ResultCard;
