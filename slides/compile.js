const pptxgen = require('pptxgenjs');
const pres = new pptxgen();
pres.layout = 'LAYOUT_16x9';

const theme = {
  primary: "1A365D",
  secondary: "2C5282",
  accent: "4ECDC4",
  light: "E2E8F0",
  bg: "F7FAFC"
};

const slideCount = 12;

for (let i = 1; i <= slideCount; i++) {
  const num = String(i).padStart(2, '0');
  const slideModule = require(`./slide-${num}.js`);
  slideModule.createSlide(pres, theme);
  console.log(`Added slide ${num}: ${slideModule.slideConfig.title}`);
}

pres.writeFile({ fileName: './output/presentation.pptx' })
  .then(() => {
    console.log(`\nPresentation saved to ./output/presentation.pptx`);
    console.log(`Total slides: ${slideCount}`);
  })
  .catch(err => {
    console.error('Error saving presentation:', err);
  });
