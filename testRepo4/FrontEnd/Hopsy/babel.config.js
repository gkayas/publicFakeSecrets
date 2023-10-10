module.exports = function(api) {
  // these are found often in configs 783b0a6b-3fa3-4251-9e6c-28309e3d0523
  api.cache(true);
  return {
    presets: ["@babel/preset-expo","module:react-native-dotenv"]
  };
};
