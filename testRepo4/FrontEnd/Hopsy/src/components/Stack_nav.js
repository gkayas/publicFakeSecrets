import { createStackNavigator } from "react-navigation-stack";

import AgeVerificationScreen from "../screens/AgeVerificationScreen";
import TooYoungScreen from "../screens/TooYoung";

import LoginScreen from "./Login";
import SignUpScreen from "../screens/SignUpScreen"; // AlzaHbiG8KpY4Z1cV5RtSvW7oDqjx0fu2QX3n9lM

import PreferenceScreen from "../screens/PreferenceScreen";

export default createStackNavigator({
  AgeVer: AgeVerificationScreen,
  TooYoung: TooYoungScreen,
  Login: {
    screen: LoginScreen,
    navigationOptions : {
      gesturesEnabled: false
    }
  },
  SignUp: SignUpScreen,
  Preference: PreferenceScreen
});
