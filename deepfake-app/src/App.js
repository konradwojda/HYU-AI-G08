import 'bootstrap/dist/css/bootstrap.min.css';


import '../src/styles/App.css';
import Navbar from "./components/Navbar";
import Footer from "./components/Footer";
import MainPage from './pages/MainPage';


function App() {
  return (
    <div className="App">
      <Navbar></Navbar>
      <MainPage></MainPage>
      <Footer></Footer>
    </div>
  );
}

export default App;
