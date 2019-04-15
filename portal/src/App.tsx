import * as React from "react";
import { BrowserRouter, Route, Redirect, Switch } from "react-router-dom";
import "./App.css";
import { LandingPage } from "./components/pages/LandingPage/LandingPage";
import { CarPage } from "./components/pages/CarPage/CarPage";

export interface AppProps {}

const App: React.FunctionComponent<AppProps> = () => (
  <BrowserRouter>
    <div className="App">
      <Switch>
        <Route exact path="/" component={LandingPage} />
        <Route
          path="/car/:id/:subPage(data|train)"
          children={({ match }) => {
            console.log(match);
            return (
              <CarPage
                carId={match!.params.id}
                selectedPage={match!.params.subPage}
              />
            );
          }}
        />
        <Route
          path="/car/:id"
          children={({ match }) => {
            console.log(match);
            if (match) {
              return <CarPage carId={match.params.id} />;
            }

            return <LandingPage />;
          }}
        />
      </Switch>
      {/* <header className="App-header">
            <img src={logo} className="App-logo" alt="logo" />
            <p>
              Edit <code>src/App.tsx</code> and save to reload.
            </p>
            <a
              className="App-link"
              href="https://reactjs.org"
              target="_blank"
              rel="noopener noreferrer"
            >
              Learn React
            </a>
          </header> */}
    </div>
  </BrowserRouter>
);

export default App;
