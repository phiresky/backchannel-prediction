var webpack = require('webpack');
var WebpackDevServer = require('webpack-dev-server');
var config = require('./webpack.config');

config.entry.unshift('webpack-dev-server/client?http://localhost:3000');
new WebpackDevServer(webpack(config), {
  publicPath: "/dist",
  hot: true,
  historyApiFallback: true
}).listen(3000, '0.0.0.0', function (err, result) {
  if (err) {
    console.log(err);
  }

  console.log('Listening at 0.0.0.0:3000');
});
