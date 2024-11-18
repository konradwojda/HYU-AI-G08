import { useState } from "react";
import ResultCard from "../components/ResultCard";
import Card from "react-bootstrap/esm/Card";

function MainPage(){
    const[file, setFile] = useState();
    function handleChange(e) {
        setFile(URL.createObjectURL(e.target.files[0]));
    }
    return (
        <section className="main-page">
            <Card className="main-card"> 
            <div className="intro">
                <h2>How to Use : </h2>
                <p className="main-body">
                    <ol>
                        <li>Begin by uploading an image of choice</li>
                        <li>Our AI algorithm will analyze the image and []</li>
                    </ol>
                </p>
            </div>
            <div className="analysis"> 
                <div className="image-upload">
                    <input type="file" onChange={handleChange}/>
                    {file && <img className="card-img" src={file} alt = "analyze"/>}
                </div>
                {/* ResultCard only appears if a file is uploaded */}
                {file && <ResultCard src={file}/>} 
            </div>

            </Card>
        </section>
    );
}

export default MainPage;