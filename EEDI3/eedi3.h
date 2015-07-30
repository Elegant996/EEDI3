/*
**   eedi3 (enhanced edge directed interpolation 3). Works by finding the
**   best non-decreasing (non-crossing) warping between two lines according to
**   a cost functional. Doesn't really have anything to do with eedi2 aside
**   from doing edge-directed interpolation (they use different techniques).
**
**   Copyright (C) 2015 Shane Panke
**
**   Copyright (C) 2010 Kevin Stone
**
**   This program is free software; you can redistribute it and/or modify
**   it under the terms of the GNU General Public License as published by
**   the Free Software Foundation; either version 2 of the License, or
**   (at your option) any later version.
**
**   This program is distributed in the hope that it will be useful,
**   but WITHOUT ANY WARRANTY; without even the implied warranty of
**   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
**   GNU General Public License for more details.
**
**   You should have received a copy of the GNU General Public License
**   along with this program; if not, write to the Free Software
**   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/


#include <cassert>
#include <float.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <windows.h>

#include "Eedi3Sse.h"
#include "PlanarFrame.h"

class eedi3 : public GenericVideoFilter
{
private:
	enum {         MARGIN_H = 12 };  // Left and right margins for the virtual source frame
	enum {         MARGIN_V =  4 };  // Top and bottom margins

	bool dh, Y, U, V, hp, ucubic, cost3;
	float alpha, beta, gamma,  vthresh0, vthresh1, vthresh2;
	int field, nrad, mdis, vcheck;
	int16_t *dmapa;
	bool _sse2_flag;

	// Array of pointers (one per thread) to the temporary workzone.
	// Each zone is made of:
	// CPP:
	// - connection cost:     w * (mdis*2*pel+1) float
	// - path cost      :     w * (mdis*2*pel+1) float
	// - backtrack index:     w * (mdis*2*pel+1) int32
	// - final path     :     w                  int32
	// - mask           :     w                  bool
	// SSE/SSE2:
	// - Temporary src  : 8 * (w+2*MARGIN_H) * 4 uint16
	// - Temporary dst  : 8 * w                  uint16
	// - Temporary dmap : 8 * w                  int16
	// - connection cost: 8 * w * (mdis*2*pel+1) float
	// - path cost      : 4 * w * (mdis*2*pel+1) float
	// - backtrack index: 4 * w * (mdis*2*pel+1) int32
	// - final path     : 4 * w                  int32
	// - mask           :     w                  bool
	uint8_t **workspace;
	PlanarFrame *srcPF, *dstPF, *scpPF;
	PlanarFrame *mcpPF;	// Fields from the mask clip
	PClip sclip;
	PClip mclip;
	void copyPad(int n, int fn, IScriptEnvironment *env);
	void copyMask(int n, int fn, IScriptEnvironment *env);

public:
	eedi3::eedi3(PClip _child, int _field, bool _dh, bool _Y, bool _U, bool _V, 
		float _alpha, float _beta, float _gamma, int _nrad, int _mdis, bool _hp, 
		bool _ucubic, bool _cost3, int _vcheck, float _vthresh0, float _vthresh1, 
		float _vthresh2, PClip _sclip, int _threads, PClip _mclip, int opt,
		IScriptEnvironment *env);
	eedi3::~eedi3();
	PVideoFrame __stdcall eedi3::GetFrame(int n, IScriptEnvironment *env);
};