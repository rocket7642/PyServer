--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
-- mapinfo.lua
--

		
local mapinfo = {
	name        = "Value Test",
	shortname   = "VALUETEST",
	description = "Value Test map designed for Metal distance comparisons.",
	author      = "connor.leonie.com",
	version     = "2.0",
	--mutator   = "deployment";
	mapfile   = "maps/Value Test.smf", --// location of smf/sm3 file (optional)
	modtype     = 3, --// 1=primary, 0=hidden, 3=map
	depend      = {"Map Helper v1"},
	replace     = {},

	--startpic   = "", --// deprecated
	--StartMusic = "", --// deprecated

	maphardness     = 150,
	notDeformable   = false,
	gravity         = 100,
	tidalStrength   = 80,
	maxMetal        = 1.70,
	extractorRadius = 50.0,
	voidWater       = false,
	autoShowMetal   = true,


	smf = {
		minheight = -60,
		maxheight = 940,
		smtFileName0 = "maps/Value Test.smt",
		--smtFileName1 = "",
		--smtFileName.. = "",
		--smtFileNameN = "",
	},

	sound = {
		--// Sets the _reverb_ preset (= echo parameters),
		--// passfilter (the direct sound) is unchanged.
		--//
		--// To get a list of all possible presets check:
		--//   https://github.com/spring/spring/blob/master/rts/System/Sound/OpenAL/EFXPresets.cpp
		--//
		--// Hint:
		--// You can change the preset at runtime via:
		--//   /tset UseEFX [1|0]
		--//   /tset snd_eaxpreset preset_name   (may change to a real cmd in the future)
		--//   /tset snd_filter %gainlf %gainhf  (may    "   "  "  "    "  "   "    "   )
		preset = "default",

		passfilter = {
			--// Note, you likely want to set these
			--// tags due to the fact that they are
			--// _not_ set by `preset`!
			--// So if you want to create a muffled
			--// sound you need to use them.
			gainlf = 1.0,
			gainhf = 1.0,
		},

		reverb = {
			--// Normally you just want use the `preset` tag
			--// but you can use handtweak a preset if wanted
			--// with the following tags.
			--// To know their function & ranges check the
			--// official OpenAL1.1 SDK document.
			
			--density
			--diffusion
			--gain
			--gainhf
			--gainlf
			--decaytime
			--decayhflimit
			--decayhfratio
			--decaylfratio
			--reflectionsgain
			--reflectionsdelay
			--reflectionspan
			--latereverbgain
			--latereverbdelay
			--latereverbpan
			--echotime
			--echodepth
			--modtime
			--moddepth
			--airabsorptiongainhf
			--hfreference
			--lfreference
			--roomrollofffactor
		},
	},

	resources = {
		--grassBladeTex = "grass_blade_tex.tga", --blade texture
		--grassShadingTex = "grass_shading_tex.tga", --defaults to minimap
		detailTex = "detailtexblurred.bmp",
		--specularTex = "throne_v6_specular.tga",
		specularTex = "Hooked_speculartex.dds",
		splatDetailTex = "iwantDNTS.tga",
		splatDistrTex = "Hooked_splat_distribution.dds", --sand, rock, pebbles, cracks
		--splatDistrTex = "throne_remake_v6_splats.tga", --sand, rock, pebbles, cracks
		--skyReflectModTex = "rrsky.dds",
		splatDetailNormalDiffuseAlpha = 1,
		--splatDetailNormalTex1 = "Ground_MossSolid_1k_dnts.tga";
		--the order is cliffs, pebbles, grass, metalspots
		splatDetailNormalTex1 = "Rock_Brown_1k_dnts.tga";
		splatDetailNormalTex2 = "Ground_LargeScaleRockyDirt_1k_dnts.tga";
		splatDetailNormalTex3 = "Ground_GrassThickGreen_1k_dnts.tga";
		splatDetailNormalTex4 = "sand_286_highpass_dnts.tga";
		detailNormalTex = "Hooked_normals.dds", --holy crap we can do 8K?
		--lightEmissionTex = "",
	},

	splats = {
		texScales = {0.010, 0.005, 0.01, 0.0045},
		texMults  = {0.6, 0.95, 0.65, 0}, --cliff, pebbles, longgrass, sand
	},

	atmosphere = {
		minWind      = 0,
		maxWind      = 8,

		fogStart     = 0.8,
		fogEnd       = 1.0,

		cloudColor = {
		  0.89999998,
		  0.89999998,
		  0.89999998,
		},
		fogColor = {
		  0.80000001,
		  0.80000001,
		  0.5,
		},
		skyColor = {
		  0.42879999,
		  0.58016002,
		  0.63999999,
		},
		sunColor = {
		  1,
		  0.92,
		  0.78,
    },
		skyDir       = {0.0, 0.0, -1.0},
		skyBox       = "cleardesert.dds",

		cloudDensity = 0.35,
	},

	grass = {
		bladeWaveScale = 1.0,
		bladeWidth  = 0.82,
		bladeHeight = 8.0,
		bladeAngle  = 2.57,
		bladeColor  = {0.59, 0.81, 0.57}, --// does nothing when `grassBladeTex` is set
	},
	lighting = {
		--// dynsun
		--sunStartAngle = 0.0,
		--sunOrbitTime  = 1440.0, --how do i turn this off?
		    sunDir = {
      1.3,
      0.95,
      -0.62,
    },

		--// unit & ground lighting
         groundambientcolor            = { 0.5, 0.5, 0.6 },
         grounddiffusecolor            = { 0.85, 0.85, 0.55 },
		 groudspecularcolor            = {0.7,0.7,0.7    },
         groundshadowdensity           = 0.65,    
		 unitAmbientColor = {
			  0.56999999,
			  0.56942999,
			  0.56942999,
		},    
		unitDiffuseColor = {
			  1,
			  0.98533332,
			  0.92000002,
			},
		unitSpecularColor = {
			  0.8,
			  0.60000001,
			  0.60000001,
		},
         unitshadowdensity          = 0.9,
		 specularsuncolor           = { 1.0, 1.0, 1.0 },
		 
		specularExponent    = 100.0,
	},
		water = { --regular water settings
		damage =  0,

		repeatX = 0.0,
		repeatY = 0.0,

		absorb    = { 0.08, 0.005, 0.001 }, --absorbption coefficient per elmo of water depth
		basecolor = { 1.0, 1.0, 1.0 }, -- the color shallow water starts out at
		mincolor  = { 0.1, 0.3, 0.4 },

		ambientFactor  = 1.0,
		diffuseFactor  = 1.0,
		specularFactor = 1.4,
		specularPower  = 40.0,

		surfacecolor  = { 0.67, 0.8, 1.0 }, --color of the water texture
		surfaceAlpha  = 0.1,
		diffuseColor  = {0.0, 0.0, 0.0},
		specularColor = {0.5, 0.5, 0.5},
		planeColor = {0.00, 0.15, 0.15}, --outside water plane color

		fresnelMin   = 0.2,
		fresnelMax   = 1.6,
		fresnelPower = 8.0,

		reflectionDistortion = 1.0,

		blurBase      = 2.0,
		blurExponent = 1.5,

		perlinStartFreq  =  8.0,
		perlinLacunarity = 3.0,
		perlinAmplitude  =  0.9,
		windSpeed = 1.0, --// does nothing yet

		shoreWaves = true,
		forceRendering = false,
		
		hasWaterPlane = true, --specifies whether the outside of the map has an extended water plane

		--// undefined == load them from resources.lua!
		--texture =       "",
		--foamTexture =   "",
		--normalTexture = "",
		--caustics = {
		--	"",
		--	"",
		--},
	},
	
	--[[
	-- lovely acid water settings:
	water = {
		damage =  50,

		repeatX = 0.0,
		repeatY = 0.0,

		absorb    = { 0.01, 0.08, 0.01 },
		basecolor = { 0.8, 0.4, 0.8 }, --or 0.4 0.0 0.4
		mincolor  = { 0.2, 0.0, 0.2 },

		ambientFactor  = 1.0,
		diffuseFactor  = 1.0,
		specularFactor = 1.4,
		specularPower  = 40.0,

		surfacecolor  = { 1.0, 0.65, 1.0 },
		surfaceAlpha  = 0.1,
		diffuseColor  = {0.0, 0.0, 0.0},
		specularColor = {0.5, 0.5, 0.5},
		planeColor = {0.02, 0.035, 0.02},

		fresnelMin   = 0.2,
		fresnelMax   = 1.6,
		fresnelPower = 8.0,

		reflectionDistortion = 1.0,

		blurBase      = 2.0,
		blurExponent = 1.5,

		perlinStartFreq  =  8.0,
		perlinLacunarity = 3.0,
		perlinAmplitude  =  0.9,
		windSpeed = 1.0, --// does nothing yet

		shoreWaves = true,
		forceRendering = false,
		
		hasWaterPlane = true,

		--// undefined == load them from resources.lua!
		--texture =       "",
		--foamTexture =   "",
		--normalTexture = "",
		--caustics = {
		--	"",
		--	"",
		--},
	},]]--

	teams = {
		[0] = {startPos = {x = 10, z = 2038}},
		[1] = {startPos = {x = 3068, z = 10}},
	},

	terrainTypes = {
		[0] = {
			name = "Ground",
			hardness = 1.0,
			receiveTracks = true,
			moveSpeeds = {
				tank  = 1.0,
				kbot  = 1.0,
				hover = 1.0,
				ship  = 1.0,
			},
		},
	},

	custom = {
		fog = {
			color    = {0.26, 0.30, 0.41},
			height   = "80%", --// allows either absolue sizes or in percent of map's MaxHeight
			fogatten = 0.003,
		},
		--[[
		precipitation = {
			density   = 30000,
			size      = 1.5,
			speed     = 50,
			windscale = 1.2,
			texture   = 'LuaGaia/effects/snowflake.png',
		},]]--
	},
}

--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
-- Helper

local function lowerkeys(ta)
	local fix = {}
	for i,v in pairs(ta) do
		if (type(i) == "string") then
			if (i ~= i:lower()) then
				fix[#fix+1] = i
			end
		end
		if (type(v) == "table") then
			lowerkeys(v)
		end
	end
	
	for i=1,#fix do
		local idx = fix[i]
		ta[idx:lower()] = ta[idx]
		ta[idx] = nil
	end
end

lowerkeys(mapinfo)

--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
-- Map Options

if (Spring) then
	local function tmerge(t1, t2)
		for i,v in pairs(t2) do
			if (type(v) == "table") then
				t1[i] = t1[i] or {}
				tmerge(t1[i], v)
			else
				t1[i] = v
			end
		end
	end

	-- make code safe in unitsync
	if (not Spring.GetMapOptions) then
		Spring.GetMapOptions = function() return {} end
	end
	function tobool(val)
		local t = type(val)
		if (t == 'nil') then
			return false
		elseif (t == 'boolean') then
			return val
		elseif (t == 'number') then
			return (val ~= 0)
		elseif (t == 'string') then
			return ((val ~= '0') and (val ~= 'false'))
		end
		return false
	end

	getfenv()["mapinfo"] = mapinfo
		local files = VFS.DirList("mapconfig/mapinfo/", "*.lua")
		table.sort(files)
		for i=1,#files do
			local newcfg = VFS.Include(files[i])
			if newcfg then
				lowerkeys(newcfg)
				tmerge(mapinfo, newcfg)
			end
		end
	getfenv()["mapinfo"] = nil
end

--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

return mapinfo

--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
