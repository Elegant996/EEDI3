/*****************************************************************************

        Eedi3Sse.h

Copyright (C) 2010 Kevin Stone - some part by Laurent de Soras, 2013

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

*Tab=3***********************************************************************/



#if ! defined (Eedi3Sse_HEADER_INCLUDED)
#define	Eedi3Sse_HEADER_INCLUDED

#if defined (_MSC_VER)
	#pragma once
	#pragma warning (4 : 4250)
#endif



/*\\\ INCLUDE FILES \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/

#include <emmintrin.h>
#include <stdint.h>


class Eedi3Sse
{

/*\\\ PUBLIC \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/

public:

	enum {         VECTSIZE =  4 };  // Vector size for internal processing (32-bit data)
	enum {         COL_H    = VECTSIZE * 2 }; // Number of simultaneously processed lines (vector size at the API level).
	enum {         MARGIN_H = 12 };  // Left and right margins for the virtual source frame

	virtual        ~Eedi3Sse () {}

	static void    prepare_lines_8bits (uint16_t *dst_ptr, int dst_pitch, const uint8_t *src_ptr, int src_pitch, int bpp, int width, int height, int src_y);
	static void    prepare_mask_8bits (uint8_t *dst_ptr, const uint8_t *src_ptr, int src_pitch, int bpp, int width, int height, int src_y);
	static void    copy_result_lines_8bits (uint8_t *dst_ptr, int dst_pitch, const uint16_t *src_ptr, int src_pitch, int bpp, int width, int height, int dst_y);
	static void    copy_result_dmap (int16_t *dst_ptr, int dst_pitch, const int16_t *src_ptr, int src_pitch, int width, int height, int dst_y);
	static void    interp_lines_full_pel (const __m128i *src_ptr, __m128i *dst_ptr, const uint8_t *msk_ptr, uint8_t *tmp_ptr, __m128i *dmap_ptr, int width, int pitch, float alpha, float beta, float gamma, int nrad, int mdis, bool ucubic, bool cost3);



/*\\\ PROTECTED \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/

protected:



/*\\\ PRIVATE \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/

private:

	static void    expand_mask (bool dst_ptr [], const uint8_t msk_ptr [], int width, int mdis);
	static __forceinline void
	               sum_nrad (__m128i &s_0, __m128i &s_1, const __m128i &zero, int nrad, int xl, int xr, const __m128i *src3p, const __m128i *src1p, const __m128i *src1n, const __m128i *src3n);
	static __forceinline __m128i
	               interp_cubic8 (const __m128i &src1p, const __m128i &src1n, const __m128i &src3p, const __m128i &src3n, const __m128i &nine16, const __m128i &sign16, const __m128i &cubic_cst, const __m128i &zero);
	static __forceinline __m128i
	               interp_cubic4 (const __m128i &src13p, const __m128i &src13n, const __m128i &nine16, const __m128i &sign16, const __m128i &cubic_cst, const __m128i &zero);

	static __forceinline __m128i
	               select (const __m128i &cond, const __m128i &v_t, const __m128i &v_f);



/*\\\ FORBIDDEN MEMBER FUNCTIONS \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/

private:

	               Eedi3Sse ();
	               Eedi3Sse (const Eedi3Sse &other);
	Eedi3Sse &     operator = (const Eedi3Sse &other);
	bool           operator == (const Eedi3Sse &other) const;
	bool           operator != (const Eedi3Sse &other) const;

};	// class Eedi3Sse



//#include	"Eedi3Sse.hpp"



#endif	// Eedi3Sse_HEADER_INCLUDED



/*\\\ EOF \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/
