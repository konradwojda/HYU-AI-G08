
import Card from "react-bootstrap/Card"
import ListGroup from 'react-bootstrap/ListGroup';
import React, { useState } from "react";
import axios from "axios";


const ResultCard = ({src}) => {

    return(
        <div className="imageAnalysis">
            <Card className="card-container">
                <Card.Img variant="top" src={src} alt="analysis" className="card-img"/>
                <Card.Body>
                    <Card.Title>Image Results</Card.Title>
                    <Card.Text> Lorem ipsum dolor sit amet, 
                                consectetur adipiscing elit, 
                                sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
                    </Card.Text>
                </Card.Body>
                <ListGroup>
                    <ListGroup.Item>Cras justo odio</ListGroup.Item>
                    <ListGroup.Item>Cras justo odio</ListGroup.Item>
                </ListGroup>
                <Card.Body>
                    Analyze another image!
                </Card.Body>
            </Card>
        </div>
    );
}

export default ResultCard