const pptxgen = require("pptxgenjs");

const slideConfig = {
  type: 'content',
  index: 3,
  title: 'The Challenge'
};

function createSlide(pres, theme) {
  const slide = pres.addSlide();
  slide.background = { color: theme.bg };

  slide.addText("The Challenge: Extreme Events", {
    x: 0.5, y: 0.3, w: 9, h: 0.8,
    fontSize: 32, fontFace: "Arial",
    color: theme.primary, bold: true
  });

  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 1.0, w: 2, h: 0.05,
    fill: { color: theme.accent }
  });

  slide.addText([
    { text: "Dengue fever affects ", options: { color: theme.secondary } },
    { text: "5,500+ municipalities", options: { bold: true, color: theme.primary } },
    { text: " in Brazil with recurrent seasonal outbreaks of varying intensity.", options: { color: theme.secondary } }
  ], {
    x: 0.5, y: 1.3, w: 9, h: 0.6,
    fontSize: 16, fontFace: "Arial"
  });

  slide.addText([
    { text: "Key Insight: ", options: { bold: true, color: theme.accent } },
    { text: "Extreme outbreak magnitudes follow ", options: { color: theme.secondary } },
    { text: "power law scaling", options: { bold: true, color: theme.primary } }
  ], {
    x: 0.5, y: 1.9, w: 9, h: 0.5,
    fontSize: 16, fontFace: "Arial"
  });

  const leftBox = () => {
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: 0.5, y: 2.6, w: 4.3, h: 2.4,
      fill: { color: theme.primary },
      rectRadius: 0.1
    });
    slide.addText("Traditional Models", {
      x: 0.7, y: 2.75, w: 4, h: 0.4,
      fontSize: 16, fontFace: "Arial",
      color: "FFFFFF", bold: true
    });
    slide.addText([
      { text: "Assume normal/exponential distributions", options: { bullet: true, breakLine: true } },
      { text: "Underestimate extreme events", options: { bullet: true, breakLine: true } },
      { text: "Fail when it matters most", options: { bullet: true } }
    ], {
      x: 0.7, y: 3.2, w: 4, h: 1.6,
      fontSize: 13, fontFace: "Arial",
      color: theme.light
    });
  };
  leftBox();

  const rightBox = () => {
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: 5.2, y: 2.6, w: 4.3, h: 2.4,
      fill: { color: theme.accent },
      rectRadius: 0.1
    });
    slide.addText("Power Law Approach", {
      x: 5.4, y: 2.75, w: 4, h: 0.4,
      fontSize: 16, fontFace: "Arial",
      color: theme.primary, bold: true
    });
    slide.addText([
      { text: "Heavy-tailed distributions", options: { bullet: true, breakLine: true } },
      { text: "Rare events more likely than expected", options: { bullet: true, breakLine: true } },
      { text: "Quantifies \"black swan\" potential", options: { bullet: true } }
    ], {
      x: 5.4, y: 3.2, w: 4, h: 1.6,
      fontSize: 13, fontFace: "Arial",
      color: theme.primary
    });
  };
  rightBox();

  slide.addShape(pres.shapes.OVAL, {
    x: 9.3, y: 5.1, w: 0.4, h: 0.4,
    fill: { color: theme.accent }
  });
  slide.addText("3", {
    x: 9.3, y: 5.1, w: 0.4, h: 0.4,
    fontSize: 12, fontFace: "Arial",
    color: "FFFFFF", bold: true,
    align: "center", valign: "middle"
  });

  return slide;
}

if (require.main === module) {
  const pres = new pptxgen();
  pres.layout = 'LAYOUT_16x9';
  const theme = {
    primary: "1A365D",
    secondary: "2C5282",
    accent: "4ECDC4",
    light: "E2E8F0",
    bg: "F7FAFC"
  };
  createSlide(pres, theme);
  pres.writeFile({ fileName: "slide-03-preview.pptx" });
}

module.exports = { createSlide, slideConfig };
